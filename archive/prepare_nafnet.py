"""
One-time extraction utility.
It reads external/NAFNet and overwrites nafnet.py and NAFNet_LICENSE.txt.
The pipeline uses the committed copies directly.
"""


import ast
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "external" / "NAFNet"
ARCHS = REPO / "basicsr" / "models" / "archs"


def extract_classes(path, names):
    source = path.read_text(encoding = "utf-8")
    tree = ast.parse(source)

    classes = {
        node.name: ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }

    return "\n\n".join(classes[name] for name in names)


header = (
    "# Extracted from the official megvii-research/NAFNet repository.\n"
    "# Copyright (c) 2022 megvii-model. All Rights Reserved.\n"
    "# Normalization helpers originate from its arch_util.py.\n"
    "# See NAFNet_LICENSE.txt for the repository license.\n\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n\n"
)

normalization = extract_classes(ARCHS / "arch_util.py", ["LayerNormFunction", "LayerNorm2d"],)

architecture = extract_classes(ARCHS / "NAFNet_arch.py", ["SimpleGate", "NAFBlock", "NAFNet"],)

output = ROOT / "nafnet.py"
output.write_text(header + normalization + "\n\n" + architecture + "\n", encoding="utf-8",)

shutil.copyfile(REPO / "LICENSE", ROOT / "NAFNet_LICENSE.txt")

print("Created:", output)