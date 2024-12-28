# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import numpy as np
import pickle
import torch
import torch.nn as nn


from mtr.config import cfg
from mtr.models.utils.transformer import transformer_encoder_layer, position_encoding_utils
from mtr.models.utils import common_layers, polyline_encoder
from mtr.utils import common_utils
from mtr.ops.knn import knn_utils



class MTREncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_cfg = config

        # build polyline encoders
        self.agent_polyline_encoder = self.build_polyline_encoder(
            in_channels=self.model_cfg.NUM_INPUT_ATTR_AGENT + 1,
            hidden_dim=self.model_cfg.NUM_CHANNEL_IN_MLP_AGENT,
            num_layers=self.model_cfg.NUM_LAYER_IN_MLP_AGENT,
            out_channels=self.model_cfg.D_MODEL
        )
        self.map_polyline_encoder = self.build_polyline_encoder(
            in_channels=self.model_cfg.NUM_INPUT_ATTR_MAP,
            hidden_dim=self.model_cfg.NUM_CHANNEL_IN_MLP_MAP,
            num_layers=self.model_cfg.NUM_LAYER_IN_MLP_MAP,
            num_pre_layers=self.model_cfg.NUM_LAYER_IN_PRE_MLP_MAP,
            out_channels=self.model_cfg.D_MODEL
        )

        # build transformer encoder layers
        self.use_local_attn = self.model_cfg.get('USE_LOCAL_ATTN', False)
        self_attn_layers = []
        for _ in range(self.model_cfg.NUM_ATTN_LAYERS):
            self_attn_layers.append(self.build_transformer_encoder_layer(
                d_model=self.model_cfg.D_MODEL,
                nhead=self.model_cfg.NUM_ATTN_HEAD,
                dropout=self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),
                normalize_before=False,
                use_local_attn=self.use_local_attn
            ))

        self.self_attn_layers = nn.ModuleList(self_attn_layers)
        self.num_out_channels = self.model_cfg.D_MODEL
        self.use_place_holder = self.model_cfg.get('USE_PLACE_HOLDER', False)
        self.object_type = self.model_cfg.OBJECT_TYPE


    def build_polyline_encoder(self, in_channels, hidden_dim, num_layers, num_pre_layers=1, out_channels=None):
        ret_polyline_encoder = polyline_encoder.PointNetPolylineEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_pre_layers=num_pre_layers,
            out_channels=out_channels
        )
        return ret_polyline_encoder

    def build_transformer_encoder_layer(self, d_model, nhead, dropout=0.1, normalize_before=False, use_local_attn=False):
        single_encoder_layer = transformer_encoder_layer.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout,
            normalize_before=normalize_before, use_local_attn=use_local_attn
        )
        return single_encoder_layer

    def apply_global_attn(self, x, x_mask, x_pos):
        """

        Args:
            x (batch_size, N, d_model):
            x_mask (batch_size, N):
            x_pos (batch_size, N, 3):
        """
        assert torch.all(x_mask.sum(dim=-1) > 0)

        batch_size, N, d_model = x.shape
        x_t = x.permute(1, 0, 2)
        x_mask_t = x_mask.permute(1, 0, 2)
        x_pos_t = x_pos.permute(1, 0, 2)
 
        pos_embedding = position_encoding_utils.gen_sineembed_for_position(x_pos_t, hidden_dim=d_model)

        for k in range(len(self.self_attn_layers)):
            x_t = self.self_attn_layers[k](
                src=x_t,
                src_key_padding_mask=~x_mask_t,
                pos=pos_embedding
            )
        x_out = x_t.permute(1, 0, 2)  # (batch_size, N, d_model)
        return x_out

    def apply_local_attn(self, x, x_mask, x_pos, num_of_neighbors):
        """

        Args:
            x (batch_size, N, d_model):
            x_mask (batch_size, N):
            x_pos (batch_size, N, 3):
        """
        assert torch.all(x_mask.sum(dim=-1) > 0)
        batch_size, N, d_model = x.shape

        x_stack_full = x.view(-1, d_model)  # (batch_size * N, d_model)
        x_mask_stack = x_mask.view(-1)
        x_pos_stack_full = x_pos.view(-1, 3)
        batch_idxs_full = torch.arange(batch_size).type_as(x)[:, None].repeat(1, N).view(-1).int()  # (batch_size * N)

        # filter invalid elements
        x_stack = x_stack_full[x_mask_stack]
        x_pos_stack = x_pos_stack_full[x_mask_stack]
        batch_idxs = batch_idxs_full[x_mask_stack]

        # knn
        batch_offsets = common_utils.get_batch_offsets(batch_idxs=batch_idxs, bs=batch_size).int()  # (batch_size + 1)
        batch_cnt = batch_offsets[1:] - batch_offsets[:-1]

        index_pair = knn_utils.knn_batch_mlogk(
            x_pos_stack, x_pos_stack,  batch_idxs, batch_offsets, num_of_neighbors
        )  # (num_valid_elems, K)

        # positional encoding
        pos_embedding = position_encoding_utils.gen_sineembed_for_position(x_pos_stack[None, :, 0:2], hidden_dim=d_model)[0]

        # local attn
        output = x_stack
        for k in range(len(self.self_attn_layers)):
            output = self.self_attn_layers[k](
                src=output,
                pos=pos_embedding,
                index_pair=index_pair,
                query_batch_cnt=batch_cnt,
                key_batch_cnt=batch_cnt,
                index_pair_batch=batch_idxs
            )

        ret_full_feature = torch.zeros_like(x_stack_full)  # (batch_size * N, d_model)
        ret_full_feature[x_mask_stack] = output

        ret_full_feature = ret_full_feature.view(batch_size, N, d_model)
        return ret_full_feature
    
    def build_motion_query(self, use_place_holder=False):
        """
        intention_points: list of # [64, 2], per object type
        """
        intention_points = None

        if use_place_holder:
            raise NotImplementedError
        else:
            intention_points_file = cfg.ROOT_DIR / self.model_cfg.INTENTION_POINTS_FILE
            with open(intention_points_file, 'rb') as f:
                intention_points_dict = pickle.load(f)

            intention_points = {}
            for cur_type in self.object_type:
                cur_intention_points = intention_points_dict[cur_type]
                cur_intention_points = torch.from_numpy(cur_intention_points).float().view(-1, 2).cuda()
                intention_points[cur_type] = cur_intention_points # [64, 2]

        return intention_points
    
    def get_motion_query(self, center_objects_type):
        num_center_objects = len(center_objects_type)
        if self.use_place_holder:
            raise NotImplementedError
        else:
            intention_points = torch.stack([
                self.intention_points[center_objects_type[obj_idx]]
                for obj_idx in range(num_center_objects)], dim=0)
            intention_points = intention_points.permute(1, 0, 2)  # (num_query, num_center_objects, 2)

            # intention_query = position_encoding_utils.gen_sineembed_for_position(intention_points, hidden_dim=self.d_model)
            # intention_query = self.intention_query_mlps(intention_query.view(-1, self.d_model)).view(-1, num_center_objects, self.d_model)  # (num_query, num_center_objects, C)
        # return intention_query, intention_points
        return intention_points
    
    def get_motion_query_non_center_objects(self, obj_types):
        """
        obj_types: (num_center_objects, num_objects)
        return: intention_points (num_center_objects, num_objects, num_query, 2)
        """
        num_center_objects, num_objects = obj_types.shape
        num_query = self.intention_points[list(self.intention_points.keys())[0]].shape[0]
        intention_points = torch.zeros((num_center_objects, num_objects, num_query, 2))
        if self.use_place_holder:
            raise NotImplementedError
        else:
            for i_center_object in range(num_center_objects):
                for i_object in range(num_objects):
                    if obj_types[i_center_object, i_object] not in self.intention_points and obj_types[i_center_object, i_object] == '':
                        # make all zeros
                        # cur_intention_points = torch.zeros((num_query, 2))
                        pass
                    else:
                        intention_points[i_center_object, i_object, :, :] = self.intention_points[obj_types[i_center_object, i_object]]

            # intention_query = position_encoding_utils.gen_sineembed_for_position(intention_points, hidden_dim=self.d_model)
            # intention_query = self.intention_query_mlps(intention_query.view(-1, self.d_model)).view(-1, num_center_objects, self.d_model)  # (num_query, num_center_objects, C)
        # return intention_query, intention_points
        return intention_points


    def forward(self, batch_dict):
        """
        Args:
            batch_dict: mtr/datasets/dataset.py
              input_dict:
              obj_trajs: [25, 81, 11, 29] (num_center_objects, num_objects, num_timestamps, num_attrs)
              obj_trajs_mask: [25, 81, 11] (num_center_objects, num_objects, num_timestamps)
              map_polylines: [25, 768, 20, 9] (num_center_objects, num_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
              map_polylines_mask (num_center_objects, num_polylines, num_points_each_polyline)
        """
        input_dict = batch_dict['input_dict']
        obj_trajs, obj_trajs_mask = input_dict['obj_trajs'].cuda(), input_dict['obj_trajs_mask'].cuda() 
        map_polylines, map_polylines_mask = input_dict['map_polylines'].cuda(), input_dict['map_polylines_mask'].cuda() 
        obj_types = input_dict['obj_types'] # (num_center_objects, num_objects)

        obj_trajs_last_pos = input_dict['obj_trajs_last_pos'].cuda() # (num_center_objects, num_objects, (x, y, z))
        map_polylines_center = input_dict['map_polylines_center'].cuda() 
        track_index_to_predict = input_dict['track_index_to_predict']

        assert obj_trajs_mask.dtype == torch.bool and map_polylines_mask.dtype == torch.bool

        num_center_objects, num_objects, num_timestamps, _ = obj_trajs.shape
        num_polylines = map_polylines.shape[1]

        center_objects_type = input_dict['center_objects_type'] #(num_center_objects)
        self.intention_points = self.build_motion_query(use_place_holder=False) # per object type, list of # [64, 2]
        # intention_points = self.get_motion_query(center_objects_type) # (num_query, num_center_objects, 2)
        # intention_points = intention_points.permute(1, 0, 2)  # (num_center_objects, num_query, 2) #(N, 64, 2)
        intention_points =  self.get_motion_query_non_center_objects(obj_types) # (num_center_objects, num_objects, num_query, 2)
        intention_points = intention_points.cuda()
        
        # want intention_points to be (num_center_objects, num_objects, num_query, 2)
        # Final goal nearest query (timestamp 80)
        obj_trajs_pos = obj_trajs[:, :, :, :2]  # (num_center_objects, num_objects, num_timestamps, 2) # (b, cars, t, [x, y])
        dist = (obj_trajs_pos[:, :, :, None, :] - intention_points[:, :, None, :, :]).norm(dim=-1)  # (num_center_objects, num_objects, num_timestamps, num_query)
        # ref_point_idx = dist.argmin(dim=-1)  # (num_center_objects, num_objects, num_timestamps)
        # choose random ref_point_idx out of 64 intention points
        ref_point_idx = torch.randint(0, intention_points.shape[2], (num_center_objects, num_objects, num_timestamps)).cuda()
        ref_dist = dist.gather(dim=-1, index=ref_point_idx[:, :, :, None])  # (num_center_objects, num_objects, num_timestamps, 1)
        ref_point = intention_points.gather(dim=2, index=ref_point_idx[:, :, :, None].expand(-1, -1, -1, 2))  # (num_center_objects, num_objects, num_timestamps, 2)

        # replace the first two dimensions of obj_trajs_pos with ref_point and ref_dist
        obj_trajs_ref_anchor = torch.cat([ref_point, ref_dist, obj_trajs[:, :, :, 3:]], dim=-1)

        # randomly pick a center object to zero out the ref_point and ref_dist by 5% of num_center_objects, and store the indices
        percentage_to_zero = 0.05
        num_center_objects_to_zero = int(num_center_objects * percentage_to_zero)
        num_center_objects_to_zero_indices = torch.randperm(num_center_objects)[:num_center_objects_to_zero]
        obj_trajs_ref_anchor[num_center_objects_to_zero_indices, :, :, :3] = 0
        batch_dict['input_dict']['num_center_objects_to_zero_indices'] = num_center_objects_to_zero_indices

        # apply polyline encoder, Ap = max_pool(MLP(Ain)),  Mp = max_pool(MLP(Min))
        # obj_trajs_in = torch.cat((obj_trajs, obj_trajs_mask[:, :, :, None].type_as(obj_trajs)), dim=-1)
        obj_trajs_in = torch.cat((obj_trajs_ref_anchor, obj_trajs_mask[:, :, :, None].type_as(obj_trajs)), dim=-1)
        obj_polylines_feature = self.agent_polyline_encoder(obj_trajs_in, obj_trajs_mask)  # (num_center_objects, num_objects, C)
        map_polylines_feature = self.map_polyline_encoder(map_polylines, map_polylines_mask)  # (num_center_objects, num_polylines, C)

        # apply self-attn (local attention)
        obj_valid_mask = (obj_trajs_mask.sum(dim=-1) > 0)  # (num_center_objects, num_objects)
        map_valid_mask = (map_polylines_mask.sum(dim=-1) > 0)  # (num_center_objects, num_polylines)
        # G
        global_token_feature = torch.cat((obj_polylines_feature, map_polylines_feature), dim=1) 
        global_token_mask = torch.cat((obj_valid_mask, map_valid_mask), dim=1) 
        global_token_pos = torch.cat((obj_trajs_last_pos, map_polylines_center), dim=1) 

        if self.use_local_attn:
            global_token_feature = self.apply_local_attn(
                x=global_token_feature, x_mask=global_token_mask, x_pos=global_token_pos,
                num_of_neighbors=self.model_cfg.NUM_OF_ATTN_NEIGHBORS
            )
        else:
            global_token_feature = self.apply_global_attn(
                x=global_token_feature, x_mask=global_token_mask, x_pos=global_token_pos
            )

        obj_polylines_feature = global_token_feature[:, :num_objects]
        map_polylines_feature = global_token_feature[:, num_objects:]
        assert map_polylines_feature.shape[1] == num_polylines

        # organize return features
        center_objects_feature = obj_polylines_feature[torch.arange(num_center_objects), track_index_to_predict]

        batch_dict['center_objects_feature'] = center_objects_feature
        batch_dict['obj_feature'] = obj_polylines_feature
        batch_dict['map_feature'] = map_polylines_feature
        batch_dict['obj_mask'] = obj_valid_mask
        batch_dict['map_mask'] = map_valid_mask
        batch_dict['obj_pos'] = obj_trajs_last_pos
        batch_dict['map_pos'] = map_polylines_center

        return batch_dict
