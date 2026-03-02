"""
reward/utils.py
"""
import re
from typing import List, Set, Tuple, Optional
from dataclasses import dataclass, field

ALLOWED_ACTIONS = {'ADD', 'STIR', 'WAIT', 'CONCENTRATE', 'YIELD', 'MAKESOLUTION', 'FILTER', 'WASH', 'DRYSOLUTION', 'COLLECTLAYER', 'EXTRACT', 'SETTEMP', 'REFLUX', 'RECRYSTAL', 'PHASESEPA', 'PH', 'PURIFY', 'QUENCH', 'PARTITION', 'TRITURATE', 'DRYSOLID', 'DEGAS', 'MICROWAVE', 'SONICATE'}
BARRIER_ACTIONS = {'STIR', 'WAIT', 'SETTEMP', 'REFLUX', 'MICROWAVE', 'SONICATE', 'FILTER', 'WASH', 'EXTRACT', 'PHASESEPA', 'QUENCH', 'DEGAS', 'RECRYSTAL', 'PURIFY', 'CONCENTRATE'}
BAG_ACTIONS = {'ADD', 'MAKESOLUTION'}

@dataclass
class ExperimentStep:
    raw_xml: str
    action: str
    chemicals: Set[str] = field(default_factory=set)
    time_val: Optional[float] = None
    time_type: str = 'none'
    temp_val: Optional[Tuple[float, float]] = None
    temp_type: str = 'none'
    clean_text: str = ""

    @property
    def is_valid_action(self) -> bool:
        return self.action in ALLOWED_ACTIONS

class ChemParser:
    @staticmethod
    def normalize_text(xml_text: str) -> str:
        # 移除 XML 标签和 Markdown 符号
        text = re.sub(r'<(procedure|time|temp|answer|think)>|</\1>', ' ', xml_text, flags=re.IGNORECASE)
        text = re.sub(r'\[MOL\]|\[/MOL\]|```xml|```', ' ', text, flags=re.IGNORECASE)
        # 移除动作前缀后的多余词 (可选，但为了 Flesh 相似度，建议保留除了标签外的所有词)
        return " ".join(text.split()).lower()

    @staticmethod
    def get_action(text: str) -> str:
        clean = re.sub(r'<[^>]+>', ' ', text).strip()
        if not clean: return "UNKNOWN"
        word = clean.split()[0].upper()
        return re.sub(r'[^A-Z]', '', word)

    @staticmethod
    def get_chemicals(text: str, action: str) -> Set[str]:
        mols = re.findall(r'\[MOL\]\s*(?:```)?(.*?)(?:```)?\s*\[/MOL\]', text, re.DOTALL)
        if mols: return {m.strip() for m in mols if m.strip()}
        # 启发式提取
        plain = re.sub(r'<[^>]+>', ' ', text).strip()
        content = plain[len(action):].strip() if plain.upper().startswith(action) else plain
        content = re.sub(r'\s*\(.*?\)', '', content)
        content = re.sub(r'^(with|in|to)\s+', '', content, flags=re.IGNORECASE)
        content = re.split(r'\s+(to|in|using|dropwise|over|until|under)\s+', content, flags=re.IGNORECASE)[0]
        return {content.strip()} if content.strip() else set()

    @staticmethod
    def get_time(text: str) -> Tuple[Optional[float], str]:
        m = re.search(r'<time>(.*?)</time>', text, re.IGNORECASE)
        if not m: return None, 'none'
        c = m.group(1).lower()
        num = re.search(r'(\d+(?:\.\d+)?)', c)
        if num:
            v = float(num.group(1))
            if 'h' in c: v *= 60
            elif 'day' in c: v *= 1440
            return v, 'numeric'
        return None, 'text'

    @staticmethod
    def get_temp(text: str) -> Tuple[Optional[Tuple[float, float]], str]:
        m = re.search(r'<temp>(.*?)</temp>', text, re.IGNORECASE)
        if not m: return None, 'none'
        nums = re.findall(r'(-?\d+(?:\.\d+)?)', m.group(1))
        if nums:
            v = sorted([float(n) for n in nums])
            return (v[0]-5.0, v[0]+5.0) if len(v)==1 else (v[0], v[-1]), 'numeric'
        return None, 'text'

    @staticmethod
    def extract_procedures(xml_text: str) -> List[str]:
        ans = re.search(r'<answer>(.*?)</answer>', xml_text, re.DOTALL | re.IGNORECASE)
        return re.findall(r'<procedure>(.*?)</procedure>', ans.group(1) if ans else xml_text, re.DOTALL)

    @classmethod
    def parse(cls, xml: str) -> ExperimentStep:
        act = cls.get_action(xml)
        return ExperimentStep(raw_xml=xml, action=act, chemicals=cls.get_chemicals(xml, act),
                              time_val=cls.get_time(xml)[0], time_type=cls.get_time(xml)[1],
                              temp_val=cls.get_temp(xml)[0], temp_type=cls.get_temp(xml)[1],
                              clean_text=cls.normalize_text(xml))