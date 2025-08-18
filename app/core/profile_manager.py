import yaml
from pathlib import Path
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ProfileManager:
    def __init__(self, profiles_path: Path):
        self.profiles_path = profiles_path
        self._profiles = self._load_profiles()

    def _load_profiles(self) -> Dict[str, Any]:
        """YAML 프로필 파일을 로드하고 캐시합니다."""
        try:
            with open(self.profiles_path, 'r') as f:
                profiles = yaml.safe_load(f)
                logger.info(f"성공적으로 모델 프로필을 로드했습니다: {list(profiles.keys())}")
                return profiles
        except FileNotFoundError:
            logger.error(f"프로필 파일을 찾을 수 없습니다: {self.profiles_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"프로필 파일 파싱 중 오류 발생: {e}")
            return {}

    def get_profile(self, profile_name: str) -> Dict[str, Any]:
        """지정된 이름의 프로필 설정을 반환합니다."""
        profile = self._profiles.get(profile_name)
        if not profile:
            logger.error(f"요청된 프로필 '{profile_name}'을(를) 찾을 수 없습니다.")
            raise ValueError(f"Profile '{profile_name}' not found.")
        return profile

# ProfileManager 인스턴스 생성 (싱글톤처럼 사용)
PROFILES_YAML_PATH = Path(__file__).parent.parent / "config" / "model_profiles.yaml"
profile_manager = ProfileManager(PROFILES_YAML_PATH)
