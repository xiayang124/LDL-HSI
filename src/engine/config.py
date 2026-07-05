from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import json


@dataclass
class BaseSetting:
    raw: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default) if default is not None else self.raw[key]

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.raw)
    
    def set(self, key: str, value: Any) -> None:
        self.raw[key] = value
        
    def update(self, updates: Dict[str, Any]) -> None:
        self.raw.update(updates)


@dataclass
class TrainingSetting:
    raw: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default) if default is not None else self.raw[key]

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.raw)

    def set(self, key: str, value: Any) -> None:
        self.raw[key] = value

    def update(self, updates: Dict[str, Any]) -> None:
        self.raw.update(updates)


@dataclass
class ModelSetting:
    raw: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default) if default is not None else self.raw[key]

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.raw)

    def set(self, key: str, value: Any) -> None:
        self.raw[key] = value

    def update(self, updates: Dict[str, Any]) -> None:
        self.raw.update(updates)

@dataclass
class ArgumentSetting:
    raw: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default) if default is not None else self.raw[key]

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.raw)

    def set(self, key: str, value: Any) -> None:
        self.raw[key] = value

    def update(self, updates: Dict[str, Any]) -> None:
        self.raw.update(updates)


@dataclass
class TestSetting:
    raw: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default) if default is not None else self.raw[key]

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.raw)

    def set(self, key: str, value: Any) -> None:
        self.raw[key] = value

    def update(self, updates: Dict[str, Any]) -> None:
        self.raw.update(updates)

@dataclass
class SVASetting:
    raw: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default) if default is not None else self.raw[key]

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.raw)

    def set(self, key: str, value: Any) -> None:
        self.raw[key] = value


@dataclass
class Config:
    BaseSetting: BaseSetting
    TrainingSetting: TrainingSetting
    ModelSetting: ModelSetting
    Argument: ArgumentSetting
    TestSetting: TestSetting
    SVASetting: SVASetting
    
    extras: Dict[str, Any] = field(default_factory=dict)
    raw: Optional[Dict[str, Any]] = None

    def get_extra(self, key: str, default: Any = None) -> Any:
        return self.extras.get(key, default)
    
    def deepcopy(self) -> 'Config':
        """Create a deep copy of the Config object."""
        return Config(
            BaseSetting=BaseSetting(raw=self.BaseSetting.to_dict()),
            TrainingSetting=TrainingSetting(raw=self.TrainingSetting.to_dict()),
            ModelSetting=ModelSetting(raw=self.ModelSetting.to_dict()),
            Argument=ArgumentSetting(raw=self.Argument.to_dict()),
            TestSetting=TestSetting(raw=self.TestSetting.to_dict()),
            SVASetting=SVASetting(raw=self.SVASetting.to_dict()),
            extras=dict(self.extras),
            raw=dict(self.raw) if self.raw is not None else None
        )


def load_config(path: str) -> Config:
    """Load JSON config from `path` and return a `Config` object."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Known top-level keys that we will parse into typed sections
    known_keys = {'BaseSetting', 'TrainingSetting', 'ModelSetting', 'Argument', 'TestSetting', 'SVASetting'}

    base = BaseSetting(raw=data.get('BaseSetting', {}))
    training = TrainingSetting(raw=data.get('TrainingSetting', {}))
    sva = SVASetting(raw=data.get('SVASetting', {}))
    # ModelSetting may be absent or renamed by user; preserve whatever is present under that key
    raw_model = data.get('ModelSetting') if 'ModelSetting' in data else {}
    model = ModelSetting(raw=raw_model if isinstance(raw_model, dict) else {})
    argument = ArgumentSetting(raw=data.get('Argument', {}))
    test = TestSetting(raw=data.get('TestSetting', {}))

    # Collect extras: keys user provided but we don't explicitly type
    extras = {k: v for k, v in data.items() if k not in known_keys}

    return Config(BaseSetting=base, TrainingSetting=training, ModelSetting=model, Argument=argument, TestSetting=test, SVASetting=sva, extras=extras, raw=data)
