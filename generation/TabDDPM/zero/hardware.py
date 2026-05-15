"""GPU inspection helpers mimicking libzero.hardware."""
from typing import List, Dict, Any

import torch


def get_gpus_info() -> List[Dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    gpus: List[Dict[str, Any]] = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        gpus.append(
            {
                "index": idx,
                "name": props.name,
                "total_memory": int(getattr(props, "total_memory", 0)),
                "multi_processor_count": int(getattr(props, "multi_processor_count", 0)),
                "compute_capability": f"{props.major}.{props.minor}",
            }
        )
    return gpus
