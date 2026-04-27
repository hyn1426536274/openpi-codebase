import logging
import math
from collections import defaultdict

import sentencepiece
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
import openpi.shared.download as _download


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05 or config.pi05_ki
        self.pi05_ki = config.pi05_ki
        # Ablation switches (only effective when pi05_ki=True)
        self.enable_fast_loss = getattr(config, "enable_fast_loss", True)
        self.enable_subtask_loss = getattr(config, "enable_subtask_loss", True)
        self.enable_ki_attention = getattr(config, "enable_ki_attention", True)

        # Load sentencepiece tokenizer for subtask generation (PI05_KI inference)
        if self.pi05_ki:
            sp_path = _download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
            with sp_path.open("rb") as f:
                self._sp_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
            self._eos_token_id = self._sp_tokenizer.eos_id()

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        if config.pytorch_compile_mode is not None:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        checkpoint_kwargs = {"use_reentrant": False, "preserve_rng_state": False}
        self._set_module_gradient_checkpointing(
            self.paligemma_with_expert.paligemma.language_model, enabled=True, checkpoint_kwargs=checkpoint_kwargs
        )
        self._set_module_gradient_checkpointing(
            self.paligemma_with_expert.paligemma.vision_tower, enabled=True, checkpoint_kwargs=checkpoint_kwargs
        )
        self._set_module_gradient_checkpointing(
            self.paligemma_with_expert.gemma_expert.model, enabled=True, checkpoint_kwargs=checkpoint_kwargs
        )

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self._set_module_gradient_checkpointing(self.paligemma_with_expert.paligemma.language_model, enabled=False)
        self._set_module_gradient_checkpointing(self.paligemma_with_expert.paligemma.vision_tower, enabled=False)
        self._set_module_gradient_checkpointing(self.paligemma_with_expert.gemma_expert.model, enabled=False)

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def _set_module_gradient_checkpointing(self, module, *, enabled: bool, checkpoint_kwargs: dict | None = None):
        if enabled and hasattr(module, "gradient_checkpointing_enable"):
            module.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs)
            return
        if not enabled and hasattr(module, "gradient_checkpointing_disable"):
            module.gradient_checkpointing_disable()
            return

        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = enabled

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        # Infer dtype from model weights so the mask matches query dtype (required by SDPA).
        dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        device = att_2d_masks_4d.device
        return torch.where(att_2d_masks_4d, torch.zeros(1, dtype=dtype, device=device), torch.full((1,), -2.3819763e38, dtype=dtype, device=device))

    def _token_ar_loss_from_hidden(self, hidden, tokenized_prompt, token_loss_mask):
        """Compute masked AR CE after selecting supervised positions.

        This is equivalent to computing logits for every text token and masking the
        loss afterwards, but avoids sending unsupervised prompt positions through
        the large vocabulary projection.
        """
        text_hidden = hidden[:, -tokenized_prompt.shape[1] :, :]
        pred_hidden = text_hidden[:, :-1, :]
        labels = tokenized_prompt[:, 1:]
        loss_mask = token_loss_mask[:, 1:].bool()
        selected_hidden = pred_hidden[loss_mask]
        selected_labels = labels[loss_mask]

        if selected_labels.numel() == 0:
            return selected_hidden.sum() * 0.0

        selected_logits = self._apply_checkpoint(
            lambda x: self.paligemma_with_expert.paligemma.lm_head(x),
            selected_hidden,
        )
        return F.cross_entropy(selected_logits, selected_labels)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix_images(
        self, images, img_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed only the image prefix so it can be reused across KI branches."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, image_prefix=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        if image_prefix is None:
            image_embs, image_pad_masks, image_att_masks = self.embed_prefix_images(images, img_masks)
        else:
            image_embs, image_pad_masks, image_att_masks = image_prefix

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs = torch.cat([image_embs, lang_emb], dim=1)
        pad_masks = torch.cat([image_pad_masks, lang_masks], dim=1)
        lang_att_masks = torch.zeros(
            (pad_masks.shape[0], lang_emb.shape[1]),
            dtype=image_att_masks.dtype,
            device=pad_masks.device,
        )
        att_masks = torch.cat([image_att_masks, lang_att_masks], dim=1)

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor | dict:
        """Do a full training forward pass and compute the loss.

        Returns:
          - PI0/PI05 mode: flow-matching loss tensor [B, T, D]
          - PI05_KI mode: dict with keys "action", "subtask", "fast"
        """
        loss = defaultdict(float)
        task_mode = "all"
        raw_task_mode = getattr(observation, "ki_task_mode", None)
        if raw_task_mode is not None:
            if isinstance(raw_task_mode, torch.Tensor):
                if raw_task_mode.numel() == 0:
                    raise ValueError("ki_task_mode tensor is empty.")
                task_mode_id = int(raw_task_mode.reshape(-1)[0].item())
            else:
                task_mode_id = int(raw_task_mode)
            task_mode_map = {0: "all", 1: "fast_flow", 2: "subtask"}
            if task_mode_id not in task_mode_map:
                raise ValueError(f"Unsupported ki_task_mode: {task_mode_id}")
            task_mode = task_mode_map[task_mode_id]

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=self.training
        )
        shared_image_prefix = None
        if self.pi05_ki:
            shared_image_prefix = self.embed_prefix_images(images, img_masks)

        # PI05_KI: compute language model losses first (subtask + fast AR)
        knowledge_isolation = False
        joint_fast_flow = False
        if self.pi05_ki:
            compute_subtask_loss = self.enable_subtask_loss and task_mode in ("all", "subtask")
            compute_fast_loss = self.enable_fast_loss and task_mode in ("all", "fast_flow")
            compute_action_loss = task_mode in ("all", "fast_flow")

            # KI attention switch: only pass knowledge_isolation=True when enabled
            has_lm_inputs = (
                observation.fast_tokenized_prompt is not None
                and observation.subtask_tokenized_prompt is not None
            )
            if self.enable_ki_attention and has_lm_inputs:
                knowledge_isolation = True
                joint_fast_flow = compute_fast_loss and compute_action_loss

            # Language model losses (subtask + fast): only compute when at least one is enabled
            need_lm = compute_fast_loss or compute_subtask_loss
            if need_lm and has_lm_inputs:
                loss.update(
                    self.forward_language_model(
                        observation,
                        images,
                        img_masks,
                        shared_image_prefix,
                        compute_subtask_loss=compute_subtask_loss,
                        compute_fast_loss=compute_fast_loss and not joint_fast_flow,
                    )
                )
            if not compute_action_loss:
                return loss

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        if joint_fast_flow:
            joint_lang_tokens = observation.fast_tokenized_prompt
            joint_lang_masks = observation.fast_tokenized_prompt_mask
        else:
            joint_lang_tokens = lang_tokens
            joint_lang_masks = lang_masks

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, joint_lang_tokens, joint_lang_masks, image_prefix=shared_image_prefix
        )
        if joint_fast_flow:
            fast_token_ar_mask = observation.fast_token_ar_mask.to(device=prefix_att_masks.device)
            prefix_att_masks = prefix_att_masks.clone()
            prefix_att_masks[:, -fast_token_ar_mask.shape[-1] :] = fast_token_ar_mask
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        if joint_fast_flow:
            prefix_len = prefix_pad_masks.shape[1]
            image_prefix_len = shared_image_prefix[1].shape[1] if shared_image_prefix is not None else prefix_pad_masks.shape[1] - joint_lang_tokens.shape[1]
            fast_loss_mask = observation.fast_token_loss_mask.to(device=att_2d_masks.device).bool()
            fast_answer_mask = torch.zeros_like(prefix_pad_masks, dtype=torch.bool)
            fast_answer_mask[:, image_prefix_len:] = fast_loss_mask
            att_2d_masks[:, prefix_len:, :prefix_len] &= ~fast_answer_mask[:, None, :]
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            image_prefix_valid_lens = prefix_pad_masks[:, :image_prefix_len].sum(dim=1)
            flow_prefix_offsets = image_prefix_valid_lens + lang_masks.sum(dim=1)
            suffix_position_ids = flow_prefix_offsets[:, None] + torch.cumsum(suffix_pad_masks, dim=1) - 1
            position_ids = torch.cat([prefix_position_ids, suffix_position_ids], dim=1)
        else:
            position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond, knowledge_isolation):
            # Ensure mask dtype matches embeddings (gradient checkpointing may recompute in float32)
            att_2d_masks_4d = att_2d_masks_4d.to(dtype=prefix_embs.dtype)
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                knowledge_isolation=knowledge_isolation,
            )
            return prefix_out, suffix_out

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond, knowledge_isolation
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        flow_loss = F.mse_loss(u_t, v_t, reduction="none")

        if self.pi05_ki:
            if joint_fast_flow:
                loss["fast"] = self._token_ar_loss_from_hidden(
                    prefix_out,
                    observation.fast_tokenized_prompt,
                    observation.fast_token_loss_mask,
                )
            loss["action"] = flow_loss.mean()
            return loss
        return flow_loss

    def forward_language_model(
        self,
        observation,
        images,
        img_masks,
        shared_image_prefix=None,
        *,
        compute_subtask_loss=True,
        compute_fast_loss=True,
    ) -> dict:
        """Compute autoregressive token losses (subtask + fast) for PI05_KI training.

        Both losses are cross-entropy AR losses over the respective token sequences.
        Only tokens where loss_mask=True contribute to the loss (i.e. the postfix / answer part).
        Respects self.enable_subtask_loss / self.enable_fast_loss switches — skips
        embedding, forward pass, and loss computation for disabled branches.
        """
        losses = {}

        def forward_func(prefix_embs, att_2d_masks_4d, position_ids):
            # Ensure mask dtype matches embeddings (gradient checkpointing may recompute in float32)
            att_2d_masks_4d = att_2d_masks_4d.to(dtype=prefix_embs.dtype)
            (prefix_out, _), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=False,
                adarms_cond=[None, None],
            )
            return prefix_out

        # --- Subtask AR loss ---
        if self.enable_subtask_loss and compute_subtask_loss:
            subtask_tokenized_prompt      = observation.subtask_tokenized_prompt
            subtask_tokenized_prompt_mask = observation.subtask_tokenized_prompt_mask
            subtask_token_ar_mask         = observation.subtask_token_ar_mask
            subtask_token_loss_mask       = observation.subtask_token_loss_mask

            subtask_prefix_embs, subtask_prefix_pad_masks, subtask_prefix_att_masks = self.embed_prefix(
                images,
                img_masks,
                subtask_tokenized_prompt,
                subtask_tokenized_prompt_mask,
                image_prefix=shared_image_prefix,
            )
            if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
                subtask_prefix_embs = subtask_prefix_embs.to(dtype=torch.bfloat16)

            subtask_pad_masks  = subtask_prefix_pad_masks
            subtask_att_masks  = subtask_prefix_att_masks.clone()
            subtask_att_masks[:, -subtask_token_ar_mask.shape[-1]:] = subtask_token_ar_mask
            subtask_att_2d_masks     = make_att_2d_masks(subtask_pad_masks, subtask_att_masks)
            subtask_position_ids     = torch.cumsum(subtask_pad_masks, dim=1) - 1
            subtask_att_2d_masks_4d  = self._prepare_attention_masks_4d(subtask_att_2d_masks)

            subtask_prefix_out = self._apply_checkpoint(forward_func, subtask_prefix_embs, subtask_att_2d_masks_4d, subtask_position_ids)
            losses["subtask"] = self._token_ar_loss_from_hidden(
                subtask_prefix_out,
                subtask_tokenized_prompt,
                subtask_token_loss_mask,
            )

        # --- FAST AR loss ---
        if self.enable_fast_loss and compute_fast_loss:
            fast_tokenized_prompt      = observation.fast_tokenized_prompt
            fast_tokenized_prompt_mask = observation.fast_tokenized_prompt_mask
            fast_token_ar_mask         = observation.fast_token_ar_mask
            fast_token_loss_mask       = observation.fast_token_loss_mask

            fast_prefix_embs, fast_prefix_pad_masks, fast_prefix_att_masks = self.embed_prefix(
                images,
                img_masks,
                fast_tokenized_prompt,
                fast_tokenized_prompt_mask,
                image_prefix=shared_image_prefix,
            )
            if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
                fast_prefix_embs = fast_prefix_embs.to(dtype=torch.bfloat16)

            fast_pad_masks  = fast_prefix_pad_masks
            fast_att_masks  = fast_prefix_att_masks.clone()
            fast_att_masks[:, -fast_token_ar_mask.shape[-1]:] = fast_token_ar_mask
            fast_att_2d_masks     = make_att_2d_masks(fast_pad_masks, fast_att_masks)
            fast_position_ids     = torch.cumsum(fast_pad_masks, dim=1) - 1
            fast_att_2d_masks_4d  = self._prepare_attention_masks_4d(fast_att_2d_masks)

            fast_prefix_out = self._apply_checkpoint(forward_func, fast_prefix_embs, fast_att_2d_masks_4d, fast_position_ids)
            losses["fast"] = self._token_ar_loss_from_hidden(
                fast_prefix_out,
                fast_tokenized_prompt,
                fast_token_loss_mask,
            )

        return losses

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    # ------------------------------------------------------------------
    # Subtask generation (PI05_KI inference)
    # ------------------------------------------------------------------

    def _build_subtask_prefix_tokens(self, observation) -> list[list[int]]:
        """Build the subtask generation prefix token sequences from the current observation.

        Prefer the already-tokenized KI subtask stream produced by
        ``TokenizeKIInputs``. That stream is built before
        ``PadStatesAndActions``, so it matches the training-time Task+State
        prefix exactly. If those fields are unavailable, fall back to the older
        reconstruction path.

        Returns a list (one per batch element) of token-id lists.
        """
        import numpy as np

        if (
            observation.subtask_tokenized_prompt is not None
            and observation.subtask_tokenized_prompt_mask is not None
        ):
            tokenized = observation.subtask_tokenized_prompt  # [B, L]
            prompt_masks = observation.subtask_tokenized_prompt_mask  # [B, L]
            loss_masks = observation.subtask_token_loss_mask  # [B, L] or None

            batch_size = tokenized.shape[0]
            all_prefix_tokens = []

            for b in range(batch_size):
                mask = prompt_masks[b]
                if isinstance(mask, torch.Tensor):
                    valid_len = int(mask.sum().item())
                    token_ids = tokenized[b, :valid_len].detach().cpu().tolist()
                    if loss_masks is not None:
                        loss_mask = loss_masks[b, :valid_len].detach().cpu()
                        first_loss = torch.nonzero(loss_mask, as_tuple=False)
                        if first_loss.numel() > 0:
                            token_ids = token_ids[: int(first_loss[0].item())]
                else:
                    valid_len = int(np.sum(mask))
                    token_ids = tokenized[b, :valid_len].tolist()
                    if loss_masks is not None:
                        loss_mask = np.asarray(loss_masks[b, :valid_len])
                        first_loss = np.flatnonzero(loss_mask)
                        if len(first_loss) > 0:
                            token_ids = token_ids[: int(first_loss[0])]

                if not token_ids:
                    raise ValueError("subtask_tokenized_prompt contains no valid prefix tokens.")
                all_prefix_tokens.append([int(token_id) for token_id in token_ids])

            return all_prefix_tokens

        # Fallback for older observations that do not carry the KI subtask
        # token stream. This path may be less aligned because observation.state
        # can already be padded at generation time.
        # Decode the main prompt from tokenized_prompt (recover text from token ids)
        tokenized = observation.tokenized_prompt  # [B, L]
        prompt_masks = observation.tokenized_prompt_mask  # [B, L]
        states = observation.state  # [B, S]

        batch_size = tokenized.shape[0]
        all_prefix_tokens = []

        for b in range(batch_size):
            mask = prompt_masks[b]
            if isinstance(mask, torch.Tensor):
                valid_len = int(mask.sum().item())
                token_ids = tokenized[b, :valid_len].cpu().tolist()
            else:
                valid_len = int(np.sum(mask))
                token_ids = tokenized[b, :valid_len].tolist()

            # Decode the original prompt text from PaliGemma tokens
            # The tokenized_prompt contains "Task: {text}, State: {state};\\nAction: "
            # We need to extract just the task text portion
            raw_text = self._sp_tokenizer.decode(token_ids)

            # Extract task description from the decoded text
            task_text = raw_text
            if "Task:" in raw_text:
                task_text = raw_text.split("Task:")[1]
                if ", State:" in task_text:
                    task_text = task_text.split(", State:")[0]
                task_text = task_text.strip()

            # Rebuild state string from observation.state
            state_np = states[b].detach().cpu().float().numpy() if isinstance(states[b], torch.Tensor) else states[b]
            discretized = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized))

            # Build prefix: same format as SubtaskTokenizer
            prefix = f"Task: {task_text}, State: {state_str};\nSubtask: "
            prefix_tokens = self._sp_tokenizer.encode(prefix, add_bos=True)
            all_prefix_tokens.append(prefix_tokens)

        return all_prefix_tokens

    @torch.no_grad()
    def generate_subtask(
        self,
        device: str | torch.device,
        observation,
        max_new_tokens: int = 30,
    ) -> list[str]:
        """Generate subtask descriptions autoregressively from the current observation.

        Uses only the PaliGemma backbone (no action expert). Steps:
          1. Embed images + subtask-prefix tokens  →  compute KV cache
          2. Autoregressively decode token-by-token until EOS or max_new_tokens
          3. Decode generated token ids back to text

        Args:
            device: torch device.
            observation: Observation dataclass (batched, B=1 typical at inference).
            max_new_tokens: maximum number of new tokens to generate.

        Returns:
            List of decoded subtask strings, one per batch element.
        """
        images, img_masks, _, _, _ = self._preprocess_observation(observation, train=False)

        # Build per-sample prefix token sequences
        prefix_token_lists = self._build_subtask_prefix_tokens(observation)
        batch_size = len(prefix_token_lists)

        # Pad to same length across batch
        max_prefix_len = max(len(t) for t in prefix_token_lists)
        padded_tokens = torch.zeros(batch_size, max_prefix_len, dtype=torch.long, device=device)
        padded_masks = torch.zeros(batch_size, max_prefix_len, dtype=torch.bool, device=device)
        for b, toks in enumerate(prefix_token_lists):
            padded_tokens[b, : len(toks)] = torch.tensor(toks, dtype=torch.long, device=device)
            padded_masks[b, : len(toks)] = True

        # Embed prefix: images + subtask-prefix language tokens
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, padded_tokens, padded_masks
        )
        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        # Forward through backbone to get KV cache + last hidden state
        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Get logits for the last prefix token → first generated token
        last_hidden = prefix_out[:, -1:, :]  # [B, 1, D]
        logits = self.paligemma_with_expert.paligemma.lm_head(last_hidden)  # [B, 1, V]
        next_token = logits[:, -1, :].argmax(dim=-1)  # [B]

        # Autoregressive decoding
        generated_tokens = [next_token.unsqueeze(1)]  # list of [B, 1]
        finished = next_token == self._eos_token_id  # [B]

        prefix_len = prefix_pad_masks.shape[1]

        for step in range(1, max_new_tokens):
            if finished.all():
                break

            # Embed the new token
            new_token_emb = self.paligemma_with_expert.embed_language_tokens(next_token.unsqueeze(1))
            emb_dim = new_token_emb.shape[-1]
            new_token_emb = new_token_emb * math.sqrt(emb_dim)
            if prefix_embs.dtype == torch.bfloat16:
                new_token_emb = new_token_emb.to(dtype=torch.bfloat16)

            # Position ids for the new token
            new_position_ids = torch.full(
                (batch_size, 1), prefix_len + step - 1, dtype=torch.long, device=device
            )

            # Attention mask: new token attends to all cached tokens + itself
            cache_len = prefix_len + step  # total tokens so far (including this one)
            new_att_mask = torch.zeros(batch_size, 1, 1, cache_len, device=device)

            # Forward through backbone with KV cache
            (new_out, _), past_key_values = self.paligemma_with_expert.forward(
                attention_mask=new_att_mask,
                position_ids=new_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[new_token_emb, None],
                use_cache=True,
            )

            logits = self.paligemma_with_expert.paligemma.lm_head(new_out)  # [B, 1, V]
            next_token = logits[:, -1, :].argmax(dim=-1)  # [B]

            # Mask out finished sequences (replace with EOS)
            next_token = torch.where(finished, self._eos_token_id, next_token)
            generated_tokens.append(next_token.unsqueeze(1))
            finished = finished | (next_token == self._eos_token_id)

        # Decode generated tokens to strings
        all_gen_tokens = torch.cat(generated_tokens, dim=1)  # [B, gen_len]
        results = []
        for b in range(batch_size):
            token_ids = all_gen_tokens[b].cpu().tolist()
            # Truncate at EOS
            if self._eos_token_id in token_ids:
                token_ids = token_ids[: token_ids.index(self._eos_token_id)]
            text = self._sp_tokenizer.decode(token_ids)
            results.append(text)

        return results
