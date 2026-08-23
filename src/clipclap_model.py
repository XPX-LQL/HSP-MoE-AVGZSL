# system, numpy
import os
import sys
import numpy as np
import math
from einops import rearrange, repeat
import einops
import opt_einsum
# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# user defined
from src.optimizer import SAM

torch.set_printoptions(threshold=10_000)
def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, nn.BatchNorm1d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, nn.BatchNorm1d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)





class EmbeddingNet(nn.Module):
    def __init__(self, input_size, output_size, dropout, use_bn, hidden_size=-1):
        super(EmbeddingNet, self).__init__()
        modules = []

        if hidden_size > 0:
            modules.append(nn.Linear(in_features=input_size, out_features=hidden_size))
            if use_bn:
                modules.append(nn.BatchNorm1d(num_features=hidden_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(in_features=hidden_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        else:
            modules.append(nn.Linear(in_features=input_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class AVFusionBlock(nn.Module):
    def __init__(self, video_dim, audio_dim, dropout=0.1):
        super(AVFusionBlock, self).__init__()
        self.audio_to_video = nn.Linear(audio_dim, video_dim)
        self.video_to_audio = nn.Linear(video_dim, audio_dim)
        self.video_gate = nn.Linear(video_dim * 2, video_dim)
        self.audio_gate = nn.Linear(audio_dim * 2, audio_dim)
        self.video_norm = nn.LayerNorm(video_dim)
        self.audio_norm = nn.LayerNorm(audio_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, video, audio):
        audio = audio.type(torch.float32)
        video = video.type(torch.float32)

        video_context = self.audio_to_video(audio)
        audio_context = self.video_to_audio(video)

        video_gate = torch.sigmoid(self.video_gate(torch.cat((video, video_context), dim=1)))
        audio_gate = torch.sigmoid(self.audio_gate(torch.cat((audio, audio_context), dim=1)))

        fused_video = self.video_norm(video + self.dropout(video_gate * video_context))
        fused_audio = self.audio_norm(audio + self.dropout(audio_gate * audio_context))
        return fused_video, fused_audio


class HypersphericalConsensusRouting(nn.Module):
    """Semantic-disagreement routing implemented as a hyperspherical AV-MoE."""

    def __init__(self, video_dim, audio_dim, semantic_dim, dropout=0.1):
        super(HypersphericalConsensusRouting, self).__init__()
        hidden_dim = max(semantic_dim * 2, 128)
        self.video_router_proj = nn.Linear(video_dim, semantic_dim)
        self.audio_router_proj = nn.Linear(audio_dim, semantic_dim)
        self.multimodal_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(video_dim + audio_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, semantic_dim)
            )
        ])
        self.singlemodal_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(video_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, semantic_dim)
            ),
            nn.Sequential(
                nn.Linear(audio_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, semantic_dim)
            )
        ])
        self.router = nn.Sequential(
            nn.Linear(semantic_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, len(self.multimodal_experts) + len(self.singlemodal_experts))
        )

    def forward(self, student_direction, video, audio, mix_weight=0.5, temperature=1.0):
        video = video.type(torch.float32)
        audio = audio.type(torch.float32)
        student_direction = F.normalize(student_direction, dim=1)

        video_descriptor = F.normalize(self.video_router_proj(video), dim=1)
        audio_descriptor = F.normalize(self.audio_router_proj(audio), dim=1)

        routing_temperature = max(float(temperature), 1e-6)
        router_input = torch.cat((video_descriptor, audio_descriptor), dim=1)
        routing_logits = self.router(router_input) / routing_temperature
        routing_weights = F.softmax(routing_logits, dim=1)

        expert_outputs = []
        av_input = torch.cat((video, audio), dim=1)
        for expert in self.multimodal_experts:
            expert_outputs.append(F.normalize(expert(av_input), dim=1))
        expert_outputs.append(F.normalize(self.singlemodal_experts[0](video), dim=1))
        expert_outputs.append(F.normalize(self.singlemodal_experts[1](audio), dim=1))
        expert_directions = torch.stack(expert_outputs, dim=1)

        moe_direction = F.normalize(
            torch.sum(routing_weights.unsqueeze(-1) * expert_directions, dim=1),
            dim=1
        )

        mix_weight = min(max(float(mix_weight), 0.0), 1.0)
        refined_direction = F.normalize(
            (1.0 - mix_weight) * student_direction + mix_weight * moe_direction,
            dim=1
        )
        return refined_direction, moe_direction, routing_weights













































class ClipClap_model(nn.Module):
    def __init__(self, params_model, input_size_audio, input_size_video):
        super(ClipClap_model, self).__init__()

        print('Initializing model variables...', end='')
        # Dimension of embedding
        self.dim_out = params_model['dim_out']
        self.input_dim_audio = input_size_audio
        self.input_dim_video = input_size_video

        self.hidden_size_decoder=params_model['decoder_hidden_size']
        self.drop_proj_o=params_model['dropout_decoder']
        self.drop_proj_w=params_model['additional_dropout']
        self.reg_loss=params_model['reg_loss']
        self.cross_entropy_loss=params_model['cross_entropy_loss']
        self.hidden_size_encoder=params_model['encoder_hidden_size']
        self.drop_enc=params_model['dropout_encoder']
        self.semantic_distill_loss = params_model.get('semantic_distill_loss', False)
        self.semantic_distill_weight = params_model.get('semantic_distill_weight', 0.0)
        self.semantic_distribution_weight = params_model.get('semantic_distribution_weight', None)
        self.semantic_relation_weight = params_model.get('semantic_relation_weight', None)
        if self.semantic_distribution_weight is None:
            self.semantic_distribution_weight = self.semantic_distill_weight
        if self.semantic_relation_weight is None:
            self.semantic_relation_weight = self.semantic_distill_weight
        self.semantic_distill_mode = params_model.get('semantic_distill_mode', 'distribution')
        self.semantic_distill_temperature = params_model.get('semantic_distill_temperature', 2.0)
        self.semantic_hyperspherical = params_model.get('semantic_hyperspherical', False)
        self.semantic_hyperspherical_temperature = params_model.get('semantic_hyperspherical_temperature', 1.0)
        self.semantic_radial_mode = params_model.get('semantic_radial_mode', 'none')
        self.semantic_radial_alpha = params_model.get('semantic_radial_alpha', 0.0)
        self.semantic_hup_loss = params_model.get('semantic_hup_loss', False)
        self.semantic_hup_weight = params_model.get('semantic_hup_weight', 0.0)
        self.semantic_hup_mode = params_model.get('semantic_hup_mode', 'uniformity')
        self.semantic_hup_temperature = params_model.get('semantic_hup_temperature', 2.0)
        self.semantic_hup_margin = params_model.get('semantic_hup_margin', 0.0)
        self.semantic_hup_cross_class_only = params_model.get('semantic_hup_cross_class_only', True)
        self.semantic_consensus_routing = params_model.get('semantic_consensus_routing', False)
        self.semantic_consensus_weight = params_model.get('semantic_consensus_weight', 0.0)
        self.semantic_consensus_temperature = params_model.get('semantic_consensus_temperature', 1.0)
        self.av_fusion_module = params_model.get('av_fusion_module', False)
        self.av_fusion_dropout = params_model.get('av_fusion_dropout', 0.1)
        if self.semantic_distill_mode not in ('distribution', 'relation', 'both'):
            raise ValueError(
                "semantic_distill_mode must be one of 'distribution', 'relation', or 'both'"
            )
        if self.semantic_hup_mode not in ('uniformity', 'preserve'):
            raise ValueError(
                "semantic_hup_mode must be either 'uniformity' or 'preserve'"
            )
        if self.semantic_radial_mode not in ('none', 'norm', 'ce'):
            raise ValueError(
                "semantic_radial_mode must be one of 'none', 'norm', or 'ce'"
            )


        self.rec_loss = params_model['rec_loss']

        self.lr_scheduler = params_model['lr_scheduler']

        print('Initializing trainable models...', end='')


        self.modality = params_model['modality']
        self.word_embeddings = params_model['word_embeddings']

        if self.modality == 'audio':
            self.O_enc = EmbeddingNet(
                input_size=1024,
                output_size=512,
                dropout=0.1,
                use_bn=True
            )
            self.W_enc = EmbeddingNet(
                input_size=1024,
                output_size=512,
                dropout=0.1,
                use_bn=True
            )
        elif self.modality == 'video':
            self.O_enc = EmbeddingNet(
                input_size=512,
                output_size=512,
                dropout=0.1,
                use_bn=True
            )
            self.W_enc = EmbeddingNet(
                input_size=512,
                output_size=512,
                dropout=0.1,
                use_bn=True
            )
        else:
            self.O_enc = EmbeddingNet(
                input_size=1536,
                output_size=512,
                dropout=0.1,
                use_bn=True
            )
            w_in_dim = 1536
            if self.word_embeddings == 'wavcaps':
                w_in_dim = 1024
            elif self.word_embeddings == 'clip':
                w_in_dim = 512

            self.W_enc = EmbeddingNet(
                input_size=w_in_dim,
                output_size=512,
                dropout=0.1,
                use_bn=True
            )
            if self.av_fusion_module:
                self.av_fusion = AVFusionBlock(
                    video_dim=512,
                    audio_dim=1024,
                    dropout=self.av_fusion_dropout
                )
            if self.semantic_consensus_routing:
                self.semantic_consensus = HypersphericalConsensusRouting(
                    video_dim=512,
                    audio_dim=1024,
                    semantic_dim=self.dim_out,
                    dropout=self.av_fusion_dropout
                )




        word_embedding_dim = 512
        self.O_proj = EmbeddingNet(
            input_size=512,
            hidden_size=self.hidden_size_decoder,
            output_size=self.dim_out,
            dropout=self.drop_proj_o,
            use_bn=params_model['embeddings_batch_norm']
        )
        self.D_o = EmbeddingNet(
            input_size=self.dim_out,
            hidden_size=self.hidden_size_decoder,
            output_size=word_embedding_dim,
            dropout=self.drop_proj_o,
            use_bn=params_model['embeddings_batch_norm']
        )


        self.W_proj= EmbeddingNet(
            input_size=word_embedding_dim,
            output_size=self.dim_out,
            dropout=self.drop_proj_w,
            use_bn=params_model['embeddings_batch_norm']
        )

        self.D_w = EmbeddingNet(
            input_size=self.dim_out,
            output_size=word_embedding_dim,
            dropout=self.drop_proj_w,
            use_bn=params_model['embeddings_batch_norm']
        )









        # Optimizers
        print('Defining optimizers...', end='')
        self.lr = params_model['lr']

        optimizer = params_model['optimizer']
        self.is_sam_optim = False
        if optimizer == 'adam':
            self.optimizer_gen = optim.Adam(
                self.parameters(),
                lr=self.lr, weight_decay=1e-5
            )
            if self.lr_scheduler:
                self.scheduler_learning_rate =  optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_gen, 'max', patience=3, verbose=True)

        elif optimizer == 'adam-sam':
            self.optimizer_gen = SAM(self.parameters(), optim.Adam, lr=self.lr, weight_decay=1e-5)
            self.is_sam_optim = True
            if self.lr_scheduler:
                # lr scheduling on base optimizer
                self.scheduler_learning_rate =  optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_gen.base_optimizer, 'max', patience=3, verbose=True)
        else:
            raise NotImplementedError

        print('Done')

        # Loss function
        print('Defining losses...', end='')
        self.criterion_cyc = nn.MSELoss()
        self.criterion_cls = nn.CrossEntropyLoss()
        self.MSE_loss = nn.MSELoss()
        print('Done')

    def _select_text_embeddings(self, w):
        if self.modality == 'audio':
            return w[:, 512:]
        elif self.modality == 'video':
            return w[:, :512]
        else:
            if self.word_embeddings == 'wavcaps':
                return w[:, 512:]
            elif self.word_embeddings == 'clip':
                return w[:, :512]
        return w

    def _build_observation_input(self, audio, video):
        audio = audio.type(torch.float32)
        video = video.type(torch.float32)
        if self.modality == 'audio':
            return audio
        elif self.modality == 'video':
            return video
        return torch.cat((video, audio), dim=1)

    def _build_encoder_input(self, audio, video):
        observation_input = self._build_observation_input(audio, video)
        if self.modality != 'both' or not self.av_fusion_module:
            return observation_input, observation_input

        # Keep the raw foundation observation for teacher-style losses such as
        # distillation / HUP, while letting the adapted branch consume an
        # explicitly fused audio-visual representation.
        fused_video, fused_audio = self.av_fusion(video, audio)
        encoder_input = torch.cat((fused_video, fused_audio), dim=1)
        return observation_input, encoder_input

    def _semantic_distill_any_active(self):
        if not self.semantic_distill_loss:
            return False
        if self.semantic_distill_mode == 'distribution':
            return self.semantic_distribution_weight > 0
        if self.semantic_distill_mode == 'relation':
            return self.semantic_relation_weight > 0
        return self.semantic_distribution_weight > 0 or self.semantic_relation_weight > 0

    def _semantic_radial_scale(self, theta):
        if self.semantic_radial_mode == 'none' or self.semantic_radial_alpha <= 0:
            return 1.0

        radius = theta.norm(p=2, dim=1, keepdim=True)
        mean_radius = radius.detach().mean().clamp_min(1e-6)
        relative_radius = (radius / mean_radius).clamp(0.5, 2.0)
        alpha = min(max(float(self.semantic_radial_alpha), 0.0), 1.0)
        return 1.0 + alpha * (relative_radius - 1.0)

    def _semantic_embedding(self, theta):
        if not self.semantic_hyperspherical:
            return theta

        direction = F.normalize(theta, dim=1)
        if self.semantic_radial_mode != 'norm':
            return direction

        # Legacy angular-radial mode: the radius is mixed into the final
        # embedding itself. Keep it available for ablation, but prefer
        # radial_mode='ce' when the final GZSL distance should stay angular.
        return direction * self._semantic_radial_scale(theta)

    def _semantic_instance_embedding(self, theta_o, audio=None, video=None):
        if not self.semantic_hyperspherical:
            return theta_o

        base_direction = F.normalize(theta_o, dim=1)
        if (
                self.modality == 'both'
                and self.semantic_consensus_routing
                and hasattr(self, 'semantic_consensus')
                and audio is not None
                and video is not None
        ):
            refined_direction, _, _ = self.semantic_consensus(
                student_direction=base_direction,
                video=video,
                audio=audio,
                mix_weight=self.semantic_consensus_weight,
                temperature=self.semantic_consensus_temperature
            )
        else:
            refined_direction = base_direction

        if self.semantic_radial_mode != 'norm':
            return refined_direction

        return refined_direction * self._semantic_radial_scale(theta_o)

    def _semantic_instance_embedding_with_aux(self, theta_o, audio=None, video=None):
        if not self.semantic_hyperspherical:
            return theta_o, None, None

        base_direction = F.normalize(theta_o, dim=1)
        consensus_direction = None
        routing_weights = None
        if (
                self.modality == 'both'
                and self.semantic_consensus_routing
                and hasattr(self, 'semantic_consensus')
                and audio is not None
                and video is not None
        ):
            refined_direction, consensus_direction, routing_weights = self.semantic_consensus(
                student_direction=base_direction,
                video=video,
                audio=audio,
                mix_weight=self.semantic_consensus_weight,
                temperature=self.semantic_consensus_temperature
            )
        else:
            refined_direction = base_direction

        if self.semantic_radial_mode != 'norm':
            return refined_direction, consensus_direction, routing_weights

        semantic_embedding = refined_direction * self._semantic_radial_scale(theta_o)
        return semantic_embedding, consensus_direction, routing_weights

    def _semantic_class_embedding(self, theta_w):
        return self._semantic_embedding(theta_w)

    def _semantic_logit_temperature(self):
        if not self.semantic_hyperspherical:
            return 1.0
        return max(float(self.semantic_hyperspherical_temperature), 1e-6)

    def _apply_semantic_ce_radial_scale(self, scores, theta_o, theta_classes):
        if (
                not self.semantic_hyperspherical
                or self.semantic_radial_mode != 'ce'
                or self.semantic_radial_alpha <= 0
        ):
            return scores

        instance_scale = self._semantic_radial_scale(theta_o)
        class_scale = self._semantic_radial_scale(theta_classes)
        return scores * instance_scale * class_scale.t()

    def _compute_semantic_distribution_distill_loss(self, theta_o, class_theta, observation_input, class_embeddings):
        if class_theta is None or class_embeddings is None:
            return torch.tensor(0., device=theta_o.device)

        temperature = max(float(self.semantic_distill_temperature), 1e-6)
        student_logits = torch.matmul(
            F.normalize(theta_o, dim=1),
            F.normalize(class_theta, dim=1).t()
        ) / temperature
        with torch.no_grad():
            teacher_logits = torch.matmul(
                F.normalize(observation_input.detach(), dim=1),
                F.normalize(class_embeddings.detach(), dim=1).t()
            ) / temperature
            teacher_probs = F.softmax(teacher_logits, dim=1)

        return F.kl_div(
            F.log_softmax(student_logits, dim=1),
            teacher_probs,
            reduction='batchmean'
        ) * (temperature ** 2)

    def _compute_semantic_relation_distill_loss(self, theta_o, observation_input):
        if theta_o.shape[0] < 2:
            return torch.tensor(0., device=theta_o.device)

        student_features = F.normalize(theta_o, dim=1)
        with torch.no_grad():
            teacher_features = F.normalize(observation_input.detach(), dim=1)
            teacher_relation = torch.matmul(teacher_features, teacher_features.t())

        student_relation = torch.matmul(student_features, student_features.t())
        mask = ~torch.eye(theta_o.shape[0], dtype=torch.bool, device=theta_o.device)
        return F.mse_loss(student_relation[mask], teacher_relation[mask])

    def _compute_semantic_distill_loss(self, theta_o, class_theta, observation_input, class_embeddings):
        if self.semantic_distill_mode == 'distribution':
            if self.semantic_distribution_weight <= 0:
                return torch.tensor(0., device=theta_o.device)
            return self.semantic_distribution_weight * self._compute_semantic_distribution_distill_loss(
                theta_o=theta_o,
                class_theta=class_theta,
                observation_input=observation_input,
                class_embeddings=class_embeddings
            )
        if self.semantic_distill_mode == 'relation':
            if self.semantic_relation_weight <= 0:
                return torch.tensor(0., device=theta_o.device)
            return self.semantic_relation_weight * self._compute_semantic_relation_distill_loss(
                theta_o=theta_o,
                observation_input=observation_input
            )

        l_distill = torch.tensor(0., device=theta_o.device)
        if self.semantic_distribution_weight > 0:
            l_distribution = self._compute_semantic_distribution_distill_loss(
                theta_o=theta_o,
                class_theta=class_theta,
                observation_input=observation_input,
                class_embeddings=class_embeddings
            )
            l_distill = l_distill + self.semantic_distribution_weight * l_distribution
        if self.semantic_relation_weight > 0:
            l_relation = self._compute_semantic_relation_distill_loss(
                theta_o=theta_o,
                observation_input=observation_input
            )
            l_distill = l_distill + self.semantic_relation_weight * l_relation
        return l_distill

    def _compute_hup_uniformity(self, features, pair_mask):
        features = F.normalize(features, dim=1)
        similarities = torch.matmul(features, features.t())
        squared_distances = (2.0 - 2.0 * similarities).clamp_min(0.0)
        selected_distances = squared_distances[pair_mask]
        if selected_distances.numel() == 0:
            return None

        scale = max(float(self.semantic_hup_temperature), 1e-6)
        uniformity_terms = -scale * selected_distances
        return torch.logsumexp(uniformity_terms, dim=0) - math.log(uniformity_terms.numel())

    def _compute_hup_direct_uniformity_loss(self, student_uniformity):
        scale = max(float(self.semantic_hup_temperature), 1e-6)
        max_squared_distance = 4.0
        # Shift the log-uniformity objective to a positive range without changing its gradients.
        return (student_uniformity + scale * max_squared_distance).clamp_min(0.0)

    def _compute_semantic_hup_loss(self, theta_o, observation_input, gt_cross_entropy):
        if theta_o.shape[0] < 2:
            return torch.tensor(0., device=theta_o.device)

        pair_mask = ~torch.eye(theta_o.shape[0], dtype=torch.bool, device=theta_o.device)
        if (
                self.semantic_hup_cross_class_only
                and gt_cross_entropy is not None
                and gt_cross_entropy.numel() == theta_o.shape[0]
        ):
            labels = gt_cross_entropy.detach().view(-1)
            cross_class_mask = labels.unsqueeze(0) != labels.unsqueeze(1)
            pair_mask = pair_mask & cross_class_mask

        student_uniformity = self._compute_hup_uniformity(theta_o, pair_mask)
        if student_uniformity is None:
            return torch.tensor(0., device=theta_o.device)

        if self.semantic_hup_mode == 'uniformity':
            return self._compute_hup_direct_uniformity_loss(student_uniformity)

        with torch.no_grad():
            teacher_uniformity = self._compute_hup_uniformity(observation_input.detach(), pair_mask)
        if teacher_uniformity is None:
            return torch.tensor(0., device=theta_o.device)

        margin = max(float(self.semantic_hup_margin), 0.0)
        return F.relu(student_uniformity - teacher_uniformity - margin)

    def optimize_scheduler(self, value):
        if self.lr_scheduler:
            self.scheduler_learning_rate.step(value)

    def forward(self, a, v, w, masks, timesteps):
        observation_input, model_input = self._build_encoder_input(a, v)
        w = self._select_text_embeddings(w)


        o = self.O_enc(model_input)

        w = self.W_enc(w)



        theta_o = self.O_proj(o)


        rho_o = self.D_o(theta_o)


        theta_w = self.W_proj(w)


        rho_w=self.D_w(theta_w)

        theta_o_semantic, consensus_direction, consensus_weights = self._semantic_instance_embedding_with_aux(
            theta_o=theta_o,
            audio=a,
            video=v
        )


        output = {
            "theta_w": theta_w,
            "theta_w_semantic": self._semantic_class_embedding(theta_w),
            "w": w,
            "rho_w": rho_w,
            "theta_o": theta_o,
            "theta_o_semantic": theta_o_semantic,
            "theta_o_consensus": consensus_direction,
            "theta_o_consensus_weights": consensus_weights,
            "rho_o": rho_o,
            "model_input": observation_input,
        }


        return output


    def compute_loss(self, outputs, embeddings_crossentropy, gt_cross_entropy, use_auxiliary_losses=False):

        theta_w = outputs['theta_w']

        w = outputs['w']
        rho_w = outputs['rho_w']

        theta_o = outputs['theta_o']

        rho_o = outputs['rho_o']

        observation_input = outputs['model_input']

        device = theta_w.device
        theta_o_semantic = outputs.get('theta_o_semantic', self._semantic_instance_embedding(theta_o))
        use_semantic_distill = (
                use_auxiliary_losses and self._semantic_distill_any_active()
        )
        use_semantic_hup = (
                use_auxiliary_losses and self.semantic_hup_loss and self.semantic_hup_weight > 0
        )
        distill_needs_class_embeddings = (
                use_semantic_distill
                and self.semantic_distill_mode in ('distribution', 'both')
                and self.semantic_distribution_weight > 0
        )
        needs_class_embeddings = self.cross_entropy_loss or distill_needs_class_embeddings
        selected_embeddings_crossentropy = None
        embedding_cross_entropy = None
        embedding_cross_entropy_semantic = None
        scores = None

        if needs_class_embeddings and embeddings_crossentropy is not None:
            selected_embeddings_crossentropy = self._select_text_embeddings(embeddings_crossentropy)
            embedding_cross_entropy = self.W_proj(self.W_enc(selected_embeddings_crossentropy))
            embedding_cross_entropy_semantic = self._semantic_class_embedding(embedding_cross_entropy)
            scores = torch.matmul(
                theta_o_semantic,
                embedding_cross_entropy_semantic.t()
            )
            scores = self._apply_semantic_ce_radial_scale(
                scores=scores,
                theta_o=theta_o,
                theta_classes=embedding_cross_entropy
            ) / self._semantic_logit_temperature()

        if self.cross_entropy_loss==True:
            Cross_loss=nn.CrossEntropyLoss()
            # gt_cross_entropy = [1, 3, 2, 55, 97, 45, ...] list of gt class labels -> shape (bs,)
            l_ce=Cross_loss(scores, gt_cross_entropy)
        else:
            l_ce = torch.tensor(0., device=device)

        if self.reg_loss==True:
            l_reg = (
                self.MSE_loss(theta_o, theta_w)
            )
        else:
            l_reg = torch.tensor(0., device=device)


        if self.rec_loss == True:
            l_rec = (
                    self.MSE_loss(w, rho_o) +
                    self.MSE_loss(w, rho_w)
            )
        else:
            l_rec = torch.tensor(0., device=device)

        if use_semantic_distill:
            l_distill = self._compute_semantic_distill_loss(
                theta_o=theta_o_semantic,
                class_theta=embedding_cross_entropy_semantic,
                observation_input=observation_input,
                class_embeddings=selected_embeddings_crossentropy
            )
        else:
            l_distill = torch.tensor(0., device=device)

        if use_semantic_hup:
            l_hup = self._compute_semantic_hup_loss(
                theta_o=theta_o_semantic,
                observation_input=observation_input,
                gt_cross_entropy=gt_cross_entropy
            )
        else:
            l_hup = torch.tensor(0., device=device)

        loss_total = (
                l_rec
                + l_reg
                + l_ce
                + l_distill
                + self.semantic_hup_weight * l_hup
        )
        loss_dict = {
            "Loss/total_loss": loss_total.detach().cpu(),
            "Loss/loss_reg": l_reg.detach().cpu(),
            "Loss/loss_cmd_rec": l_rec.detach().cpu(),
            "Loss/cross_entropy": l_ce.detach().cpu(),
            "Loss/semantic_distill": l_distill.detach().cpu(),
            "Loss/semantic_hup": l_hup.detach().cpu()

        }
        return loss_total, loss_dict

    # cls_numeric = class index
    # cls_embedding = w2v embedding of the target
    def optimize_params(self, audio, video, cls_numeric, cls_embedding, masks, timesteps, embedding_crossentropy,
                        optimize=False):
        if not self.is_sam_optim:
            # Forward pass
            outputs = self.forward(audio, video, cls_embedding, masks, timesteps)

            # Backward pass
            loss_numeric, loss = self.compute_loss(
                outputs, embedding_crossentropy, cls_numeric,
                use_auxiliary_losses=optimize
            )

            if optimize == True:
                self.optimizer_gen.zero_grad()
                loss_numeric.backward()
                self.optimizer_gen.step()

        else:
            # SAM optimizer requires two forward / backward

            enable_running_stats(self)
            outputs = self.forward(audio, video, cls_embedding, masks, timesteps)
            loss_numeric, loss = self.compute_loss(
                outputs, embedding_crossentropy, cls_numeric,
                use_auxiliary_losses=optimize
            )

            if optimize:
                # first forward-backward step
                # self.optimizer_gen.zero_grad()
                loss_numeric.backward()
                self.optimizer_gen.first_step(zero_grad=True)

                # second forward-backward step
                disable_running_stats(self)
                outputs_second = self.forward(audio, video, cls_embedding, masks, timesteps)
                second_loss, _ = self.compute_loss(
                    outputs_second, embedding_crossentropy, cls_numeric,
                    use_auxiliary_losses=optimize
                )
                second_loss.backward()
                self.optimizer_gen.second_step(zero_grad=True)

        return loss_numeric, loss

    def get_embeddings(self, a, v, w, masks, timesteps):
        _, model_input = self._build_encoder_input(a, v)
        w = self._select_text_embeddings(w)


        o = self.O_enc(model_input)

        w = self.W_enc(w)



        theta_o = self.O_proj(o)

        theta_w=self.W_proj(w)

        theta_o_semantic = self._semantic_instance_embedding(theta_o, a, v)
        theta_w_semantic = self._semantic_class_embedding(theta_w)

        return theta_o_semantic, theta_o_semantic, theta_w_semantic

    def get_foundation_embeddings(self, a, v, w):
        model_input = self._build_observation_input(a, v).type(torch.float32)
        w = self._select_text_embeddings(w).type(torch.float32)
        return model_input, model_input, w
