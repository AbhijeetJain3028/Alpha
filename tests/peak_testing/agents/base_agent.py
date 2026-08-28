"""
Base Test Agent - Foundation for all test agents
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import uuid
import time
import traceback
import json
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class TestResult:
    """Result of a single test execution"""
    test_id: str
    agent_id: str
    agent_type: str
    test_name: str
    status: str  # 'passed', 'failed', 'error', 'skipped', 'partial'
    message: str
    details: Dict = None
    duration_ms: float = 0
    timestamp: str = None
    severity: str = 'info'  # 'critical', 'high', 'medium', 'low', 'info'
    reproduction_steps: List[str] = None
    expected_behavior: str = ""
    actual_behavior: str = ""
    user_impact: str = ""
    fix_suggestion: str = ""
    metadata: Dict = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
        if self.details is None:
            self.details = {}
        if self.reproduction_steps is None:
            self.reproduction_steps = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class AgentConfig:
    """Configuration for a test agent"""
    agent_id: str
    agent_type: str
    name: str
    description: str
    priority: int = 1
    enabled: bool = True
    timeout_seconds: int = 300
    retry_count: int = 3
    parallel: bool = False
    tags: List[str] = None
    config: Dict = None
    dependencies: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.config is None:
            self.config = {}
        if self.dependencies is None:
            self.dependencies = []

class BaseTestAgent(ABC):
    """Base class for all test agents"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = config.agent_id
        self.agent_type = config.agent_type
        self.name = config.name
        self.results: List[TestResult] = []
        self._lock = threading.Lock()
        self._start_time = None
        self._end_time = None
        self._error_count = 0
        self._success_count = 0
        self._skip_count = 0
        self._error_details = []
        
    @abstractmethod
    def execute(self, context: Dict[str, Any] = None) -> List[TestResult]:
        """Execute the test agent's tests"""
        pass
    
    def _create_result(self, test_name: str, status: str, message: str, 
                       details: Dict = None, severity: str = 'info',
                       expected: str = "", actual: str = "", 
                       reproduction_steps: List[str] = None,
                       user_impact: str = "", fix_suggestion: str = "") -> TestResult:
        """Create a standardized test result"""
        test_id = str(uuid.uuid4())
        return TestResult(
            test_id=test_id,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            test_name=test_name,
            status=status,
            message=message,
            details=details or {},
            severity=severity,
            expected_behavior=expected,
            actual_behavior=actual,
            reproduction_steps=reproduction_steps or [],
            user_impact=user_impact,
            fix_suggestion=fix_suggestion,
            metadata={'agent_name': self.name}
        )
    
    def add_result(self, result: TestResult):
        """Thread-safe result addition"""
        with self._lock:
            self.results.append(result)
            if result.status == 'passed':
                self._success_count += 1
            elif result.status == 'failed':
                self._error_count += 1
                self._error_details.append(result)
            elif result.status == 'skipped':
                self._skip_count += 1
    
    def run(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the agent with full lifecycle management"""
        self._start_time = time.time()
        try:
            results = self.execute(context or {})
            self._end_time = time.time()
            return self._generate_report()
        except Exception as e:
            self._end_time = time.time()
            error_result = self._create_result(
                test_name="agent_execution",
                status='error',
                message=f"Agent execution failed: {str(e)}",
                details={'traceback': traceback.format_exc()},
                severity='critical',
                user_impact="Agent completely failed to execute",
                fix_suggestion="Review agent implementation and dependencies"
            )
            self.add_result(error_result)
            return self._generate_report()
    
    def _generate_report(self) -> Dict[str, Any]:
        duration = (self._end_time - self._start_time) if self._end_time else 0
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'name': self.name,
            'duration_seconds': duration,
            'total_tests': len(self.results),
            'passed': self._success_count,
            'failed': self._error_count,
            'skipped': self._skip_count,
            'results': [r.__dict__ for r in self.results],
            'summary': self._generate_summary()
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        critical = sum(1 for r in self.results if r.severity == 'critical')
        high = sum(1 for r in self.results if r.severity == 'high')
        medium = sum(1 for r in self.results if r.severity == 'medium')
        low = sum(1 for r in self.results if r.severity == 'low')
        
        return {
            'total_issues': critical + high + medium + low,
            'by_severity': {'critical': critical, 'high': high, 'medium': medium, 'low': low},
            'success_rate': self._success_count / len(self.results) if self.results else 0,
            'critical_issues': [r for r in self.results if r.severity == 'critical'],
            'high_priority_issues': [r for r in self.results if r.severity == 'high']
        }
    
    def get_critical_failures(self) -> List[TestResult]:
        """Get all critical failures"""
        return [r for r in self.results if r.severity == 'critical']
    
    def get_all_failures(self) -> List[TestResult]:
        """Get all failures (critical, high, medium)"""
        return [r for r in self.results if r.severity in ['critical', 'high', 'medium']]


class AgentOrchestrator:
    """Orchestrates execution of multiple test agents"""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.agents: Dict[str, BaseTestAgent] = {}
        self.results: Dict[str, List[TestResult]] = {}
        self.global_results: List[TestResult] = []
        self._lock = threading.Lock()
        self._execution_order = []
        self._dependency_graph = {}
    
    def register_agent(self, agent: BaseTestAgent, dependencies: List[str] = None):
        """Register an agent for execution"""
        self.agents[agent.config.agent_id] = agent
        self._dependency_graph[agent.config.agent_id] = dependencies or []
    
    def _topological_sort(self) -> List[str]:
        """Topological sort for dependency resolution"""
        visited = set()
        temp = set()
        order = []
        
        def visit(node):
            if node in temp:
                raise ValueError(f"Circular dependency detected: {node}")
            if node not in visited:
                temp.add(node)
                for dep in self._dependency_graph.get(node, []):
                    visit(dep)
                temp.remove(node)
                visited.add(node)
                order.append(node)
        
        for node in self.agents:
            if node not in visited:
                visit(node)
        return order
    
    def run_all(self, context: Dict[str, Any] = None, 
                progress_callback: Callable = None) -> Dict[str, Any]:
        """Run all agents respecting dependencies"""
        order = self._topological_sort()
        results = {}
        
        # Execute in dependency order
        for agent_id in order:
            agent = self.agents[agent_id]
            if not agent.config.enabled:
                continue
            
            if progress_callback:
                progress_callback(agent_id, 'starting')
            
            result = agent.run(context)
            results[agent_id] = result
            self.results[agent_id] = agent.results
            self.global_results.extend(agent.results)
            
            if progress_callback:
                progress_callback(agent_id, 'completed', result)
        
        return self._generate_global_report()
    
    def run_parallel(self, context: Dict[str, Any] = None,
                     progress_callback: Callable = None) -> Dict[str, Any]:
        """Run independent agents in parallel"""
        # Group by dependency level
        levels = self._compute_execution_levels()
        
        for level_agents in levels:
            # Run agents in this level in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._run_agent, agent_id, context): agent_id
                    for agent_id in level_agents
                }
                
                for future in as_completed(futures):
                    agent_id = futures[future]
                    try:
                        result = future.result()
                        self.results[agent_id] = self.agents[agent_id].results
                        self.global_results.extend(self.agents[agent_id].results)
                    except Exception as e:
                        # Handle agent failure
                        pass
        
        return self._generate_global_report()
    
    def _compute_execution_levels(self) -> List[List[str]]:
        """Compute execution levels for parallel execution"""
        # Simplified - in reality would use topological levels
        all_agents = list(self.agents.keys())
        return [all_agents]
    
    def _run_agent(self, agent_id: str, context: Dict) -> Dict:
        agent = self.agents[agent_id]
        return agent.run(context)
    
    def _generate_global_report(self) -> Dict[str, Any]:
        total_tests = len(self.global_results)
        passed = sum(1 for r in self.global_results if r.status == 'passed')
        failed = sum(1 for r in self.global_results if r.status == 'failed')
        skipped = sum(1 for r in self.global_results if r.status == 'skipped')
        errors = sum(1 for r in self.global_results if r.status == 'error')
        
        critical = sum(1 for r in self.global_results if r.severity == 'critical')
        high = sum(1 for r in self.global_results if r.severity == 'high')
        
        return {
            'total_agents': len(self.agents),
            'total_tests': total_tests,
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
            'errors': errors,
            'critical_issues': critical,
            'high_priority_issues': high,
            'all_results': self.global_results,
            'agent_results': {k: [r.__dict__ for r in v] for k, v in self.results.items()},
            'summary': {
                'success_rate': passed / total_tests if total_tests > 0 else 0,
                'failure_rate': failed / total_tests if total_tests > 0 else 0,
                'critical_issues': critical,
                'high_priority_issues': high
            }
        }
    
    def get_all_critical_failures(self) -> List[TestResult]:
        """Get all critical failures across all agents"""
        failures = []
        for results in self.results.values():
            for r in results:
                if r.severity == 'critical':
                    failures.append(r)
        return failures
    
    def get_all_failures(self) -> List[TestResult]:
        """Get all failures across all agents"""
        failures = []
        for results in self.results.values():
            for r in results:
                if r.severity in ['critical', 'high', 'medium']:
                    failures.append(r)
        return failures
    
    def export_report(self, format: str = 'json') -> str:
        """Export global report"""
        report = self._generate_global_report()
        if format == 'json':
            return json.dumps(report, indent=2, default=str)
        elif format == 'html':
            return self._generate_html_report(report)
        return str(report)
    
    def _generate_html_report(self, report: Dict) -> str:
        """Generate HTML report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Peak Testing Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .critical {{ color: red; font-weight: bold; }}
                .high {{ color: orange; font-weight: bold; }}
                .medium {{ color: orange; }}
                .low {{ color: green; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .error {{ color: darkred; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Peak Testing Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Tests: {report['total_tests']}</p>
                <p>Passed: <span class="passed">{report['passed']}</span></p>
                <p>Failed: <span class="failed">{report['failed']}</span></p>
                <p>Critical: <span class="critical">{report['critical_issues']}</span></p>
                <p>High Priority: <span class="high">{report['high_priority_issues']}</span></p>
            </div>
        """
        return html


class TestReporter:
    """Generates detailed test reports"""
    
    @staticmethod
    def generate_detailed_report(results: List[TestResult]) -> Dict:
        """Generate detailed report from test results"""
        by_agent = {}
        by_severity = {}
        by_type = {}
        
        for r in results:
            # By agent
            if r.agent_id not in by_agent:
                by_agent[r.agent_id] = []
            by_agent[r.agent_id].append(r)
            
            # By severity
            if r.severity not in by_severity:
                by_severity[r.severity] = []
            by_severity[r.severity].append(r)
            
            # By agent type
            if r.agent_type not in by_type:
                by_type[r.agent_type] = []
            by_type[r.agent_type].append(r)
        
        return {
            'by_agent': {k: len(v) for k, v in by_agent.items()},
            'by_severity': {k: len(v) for k, v in by_severity.items()},
            'by_type': {k: len(v) for k, v in by_type.items()},
            'total': len(results),
            'details': [r.__dict__ for r in results]
        }
    
    @staticmethod
    def generate_markdown_report(results: List[TestResult]) -> str:
        """Generate markdown report"""
        md = "# Peak Testing Report\n\n"
        md += f"Generated: {datetime.utcnow().isoformat()}\n\n"
        
        # Summary
        total = len(results)
        passed = sum(1 for r in results if r.status == 'passed')
        failed = sum(1 for r in results if r.status == 'failed')
        errors = sum(1 for r in results if r.status == 'error')
        
        md += f"## Summary\n"
        md += f"- Total Tests: {total}\n"
        md += f"- Passed: {passed}\n"
        md += f"- Failed: {failed}\n"
        md += f"- Errors: {errors}\n\n"
        
        # By severity
        md += "## By Severity\n"
        for sev in ['critical', 'high', 'medium', 'low', 'info']:
            count = sum(1 for r in results if r.severity == sev)
            if count > 0:
                md += f"- {sev.capitalize()}: {count}\n"
        
        # Failed tests details
        failed_tests = [r for r in results if r.status in ['failed', 'error']]
        if failed_tests:
            md += "\n## Failed Tests\n"
            for r in failed_tests:
                md += f"\n### {r.test_name} ({r.agent_type})\n"
                md += f"- **Status**: {r.status}\n"
                md += f"- **Severity**: {r.severity}\n"
                md += f"- **Message**: {r.message}\n"
                md += f"- **Expected**: {r.expected_behavior}\n"
                md += f"- **Actual**: {r.actual_behavior}\n"
                md += f"- **Impact**: {r.user_impact}\n"
                md += f"- **Fix**: {r.fix_suggestion}\n"
                if r.reproduction_steps:
                    md += "- **Reproduction**:\n"
                    for step in r.reproduction_steps:
                        md += f"  1. {step}\n"
        
        return md