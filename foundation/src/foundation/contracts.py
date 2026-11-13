#versioned contract between decisions and execution 'services'

from dataclasses import dataclass, fields
from types import MappingProxyType
from collections.abc import Mapping
from pathlib import Path
import json
import numpy as np
from foundation.data import asof_timestamp
from foundation.store import atomic_json, fingerprint


@dataclass(frozen=True)
class TargetPositions:
    system: str
    target_id: str
    created_at: str
    effective_at: str
    expires_at: str
    currency: str
    quantities: dict[str,float]
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version!=1 or not self.system or not self.target_id or not self.currency:
            raise ValueError('Unsupported target schema or missing metadata')
        created, effective, expires = map(asof_timestamp,[self.created_at,self.effective_at,self.expires_at])
        if not created<=effective<expires:
            raise ValueError('Invalid target validity interval')
        if not isinstance(self.quantities,Mapping) or any(not k or not isinstance(k,str) for k in self.quantities) or not np.isfinite(list(self.quantities.values())).all():
            raise ValueError('Finite named target quantities required')
        
        #an alone frozen dataclass cannot freeze its 'caller-owned' dictionary
        object.__setattr__(self,'quantities',MappingProxyType(dict(self.quantities)))

    def validate_at(self,at):
        at=asof_timestamp(at)
        if not asof_timestamp(self.effective_at)<=at<asof_timestamp(self.expires_at): raise ValueError('Target not effective or expired')
        return self

    def to_dict(self):
        return {f.name:(dict(self.quantities) if f.name=="quantities" else getattr(self,f.name)) for f in fields(self)}

    def publish(self,path):
        return atomic_json(path,self.to_dict())

    @classmethod
    def load(cls,path):
        return cls(**json.loads(Path(path).read_text()))

    @property
    def sha256(self):
        return fingerprint(self.to_dict())
