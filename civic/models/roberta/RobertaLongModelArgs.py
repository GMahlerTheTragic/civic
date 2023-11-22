from dataclasses import dataclass, field


@dataclass
class RobertaLongModelArgs:
    attention_window: int = field(
        default=512, metadata={"help": "Size of attention window"}
    )
    max_pos: int = field(default=512, metadata={"help": "Maximum position"})
