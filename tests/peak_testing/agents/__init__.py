"""
Peak Testing Framework - Quantum Electron Microscopy Depth Testing
Thousands of agents testing every aspect from user perspectives.
"""

from .base_agent import BaseTestAgent
from .user_perspective_agents import *
from .edge_case_agents import *
from .stress_test_agents import *
from .security_agents import *
from .accessibility_agents import *

__all__ = [
    'BaseTestAgent',
    'UserPerspectiveAgent',
    'NewUserAgent',
    'PowerUserAgent',
    'DeveloperAgent',
    'CasualUserAgent',
    'NonTechnicalUserAgent',
    'ExpertUserAgent',
    'MobileUserAgent',
    'AccessibilityUserAgent',
    'EdgeCaseAgent',
    'EmptyInputAgent',
    'MaxLengthAgent',
    'SpecialCharactersAgent',
    'UnicodeAgent',
    'RapidFireAgent',
    'ConcurrentUserAgent',
    'LongSessionAgent',
    'MemoryPressureAgent',
    'NetworkFlakyAgent',
    'SecurityAuditAgent',
    'InjectionAttackAgent',
    'DataPrivacyAgent',
    'RateLimitAgent',
    'AccessibilityAuditAgent',
    'ScreenReaderAgent',
    'KeyboardNavigationAgent',
    'HighContrastAgent',
    'ReducedMotionAgent',
]

# Agent registry for orchestration
AGENT_REGISTRY = {}