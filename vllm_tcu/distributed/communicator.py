#
from typing import List, Optional

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.base_device_communicator import \
    DeviceCommunicatorBase


class TCUCommunicator(DeviceCommunicatorBase):

    def __init__(self,
                 cpu_group: dist.ProcessGroup,
                 device: Optional[torch.device] = None,
                 device_group: Optional[dist.ProcessGroup] = None,
                 unique_name: str = ""):
        super().__init__(cpu_group, device, device_group, unique_name)
        # TODO(hz): Refer to CudaCommunicator's implementation to integrate PyHcclCommunicator
        # init device according to rank
        self.device = torch.npu.current_device()

    def all_to_all(self,
                   input_: torch.Tensor,
                   scatter_dim: int = 0,
                   gather_dim: int = -1,
                   scatter_sizes: Optional[List[int]] = None,
                   gather_sizes: Optional[List[int]] = None) -> torch.Tensor:

        if scatter_dim < 0:
            scatter_dim += input_.dim()
        if gather_dim < 0:
            gather_dim += input_.dim()

        if scatter_sizes is not None and gather_sizes is not None:
            input_list = [
                t.contiguous()
                for t in torch.split(input_, scatter_sizes, scatter_dim)
            ]
            output_list = []
            tensor_shape_base = input_list[self.rank].size()
            for i in range(self.world_size):
                tensor_shape = list(tensor_shape_base)
                tensor_shape[gather_dim] = gather_sizes[i]
                output_list.append(
                    torch.empty(tensor_shape,
                                dtype=input_.dtype,
                                device=input_.device))

        else:
            input_list = [
                t.contiguous() for t in torch.tensor_split(
                    input_, self.world_size, scatter_dim)
            ]
            output_list = [
                torch.empty_like(input_list[i]) for i in range(self.world_size)
            ]

        dist.all_to_all(output_list, input_list, group=self.device_group)
        output_tensor = torch.cat(output_list, dim=gather_dim).contiguous()
        return output_tensor