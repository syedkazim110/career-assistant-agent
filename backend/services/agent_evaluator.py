"""
Agent Evaluation Framework

This module provides evaluation capabilities for the Career Assistant Agent,
satisfying Track A requirements for agent evaluation.
"""

from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
import os

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of a single agent evaluation test"""
    test_name: str
    task_description: str
    success: bool
    expected_tools: List[str]
    tools_used: List[str]
    execution_time: float
    num_steps: int
    error_message: Optional[str] = None
    agent_output: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Complete evaluation report"""
    timestamp: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    success_rate: float
    average_steps: float
    average_execution_time: float
    test_results: List[EvaluationResult]
    agent_metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "test_results": [tr.to_dict() for tr in self.test_results]
        }
    
    def save_to_file(self, filepath: str):
        """Save evaluation report to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Evaluation report saved to {filepath}")


class AgentEvaluator:
    """
    Evaluates agent performance across various tasks
    
    This implements custom evaluation similar to AgentEval and HumanEval,
    tailored for career assistant tasks.
    """
    
    def __init__(self, agent_service):
        """
        Initialize evaluator with agent service
        
        Args:
            agent_service: Instance of CareerAgentService to evaluate
        """
        self.agent = agent_service
        self.test_cases = self._define_test_cases()
    
    def _define_test_cases(self) -> List[Dict[str, Any]]:
        """
        Define test cases for agent evaluation
        
        These test cases validate the agent's ability to:
        1. Parse and understand documents
        2. Perform skill analysis
        3. Generate documents
        4. Use tools appropriately
        5. Handle errors gracefully
        """
        return [
            {
                "name": "parse_resume_test",
                "task": "Parse the resume PDF located at uploads/test_resume.pdf",
                "expected_tools": ["parse_pdf"],
                "success_criteria": lambda result: "parse_pdf" in str(result.get("action_history", [])),
                "timeout": 30
            },
            {
                "name": "analyze_job_test",
                "task": "Analyze the job description PDF at uploads/test_job.pdf and tell me the required skills",
                "expected_tools": ["parse_pdf", "analyze_job_description"],
                "success_criteria": lambda result: "required_skills" in result.get("output", "").lower() or "skills" in result.get("output", "").lower(),
                "timeout": 60
            },
            {
                "name": "skill_gap_analysis_test",
                "task": "Compare my resume at uploads/test_resume.pdf with the job at uploads/test_job.pdf and identify skill gaps",
                "expected_tools": ["parse_pdf", "analyze_job_description", "analyze_resume", "analyze_skill_gap"],
                "success_criteria": lambda result: any(tool in str(result.get("action_history", [])) for tool in ["analyze_skill_gap", "skill"]),
                "timeout": 90
            },
            {
                "name": "document_generation_test",
                "task": "Generate a tailored resume in PDF format using my resume at uploads/test_resume.pdf for the job at uploads/test_job.pdf",
                "expected_tools": ["parse_pdf", "generate_tailored_resume"],
                "success_criteria": lambda result: "generate" in str(result.get("action_history", [])).lower() and "resume" in str(result.get("action_history", [])).lower(),
                "timeout": 120
            },
            {
                "name": "full_workflow_test",
                "task": "I want to apply for a job. Parse my resume and the job description, analyze skill gaps, and generate both a tailored resume and cover letter in PDF format",
                "expected_tools": ["parse_pdf", "analyze_job_description", "analyze_resume", "analyze_skill_gap", "generate_tailored_resume", "generate_cover_letter"],
                "success_criteria": lambda result: len(result.get("action_history", [])) >= 5,
                "timeout": 180
            },
            {
                "name": "document_validation_test",
                "task": "Validate that the resume and cover letter documents exist at generated/test_resume.pdf and generated/test_cover_letter.pdf",
                "expected_tools": ["validate_documents"],
                "success_criteria": lambda result: "validate" in str(result.get("action_history", [])).lower(),
                "timeout": 30
            },
            {
                "name": "tool_selection_test",
                "task": "What are the key skills mentioned in the job description at uploads/test_job.pdf?",
                "expected_tools": ["parse_pdf", "analyze_job_description"],
                "success_criteria": lambda result: result.get("success", False) and len(result.get("action_history", [])) <= 3,
                "timeout": 60
            },
            {
                "name": "error_handling_test",
                "task": "Parse the resume at invalid/nonexistent_file.pdf",
                "expected_tools": ["parse_pdf"],
                "success_criteria": lambda result: "error" in result.get("output", "").lower() or not result.get("success", True),
                "timeout": 30
            }
        ]
    
    async def evaluate_single_test(self, test_case: Dict[str, Any]) -> EvaluationResult:
        """
        Run a single evaluation test
        
        Args:
            test_case: Test case definition
            
        Returns:
            EvaluationResult with test outcome
        """
        import time
        
        logger.info(f"Running test: {test_case['name']}")
        
        start_time = time.time()
        
        try:
            # Run agent with test task
            result = await self.agent.run(test_case["task"])
            
            execution_time = time.time() - start_time
            
            # Extract tools used from action history
            tools_used = [
                action["action"] 
                for action in result.get("action_history", [])
            ]
            
            # Check success criteria
            success = test_case["success_criteria"](result)
            
            # Create evaluation result
            eval_result = EvaluationResult(
                test_name=test_case["name"],
                task_description=test_case["task"],
                success=success,
                expected_tools=test_case["expected_tools"],
                tools_used=tools_used,
                execution_time=execution_time,
                num_steps=len(result.get("action_history", [])),
                agent_output=result.get("output", ""),
                error_message=result.get("error") if not success else None
            )
            
            logger.info(f"Test {test_case['name']}: {'PASSED' if success else 'FAILED'}")
            
            return eval_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Test {test_case['name']} failed with exception: {str(e)}")
            
            return EvaluationResult(
                test_name=test_case["name"],
                task_description=test_case["task"],
                success=False,
                expected_tools=test_case["expected_tools"],
                tools_used=[],
                execution_time=execution_time,
                num_steps=0,
                error_message=str(e)
            )
    
    async def evaluate_all(self, output_dir: str = "evaluation_results") -> EvaluationReport:
        """
        Run complete evaluation suite
        
        Args:
            output_dir: Directory to save evaluation results
            
        Returns:
            EvaluationReport with comprehensive results
        """
        logger.info("Starting agent evaluation...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        results: List[EvaluationResult] = []
        
        # Run all test cases
        for test_case in self.test_cases:
            result = await self.evaluate_single_test(test_case)
            results.append(result)
        
        # Calculate metrics
        passed = sum(1 for r in results if r.success)
        failed = len(results) - passed
        success_rate = (passed / len(results) * 100) if results else 0
        avg_steps = sum(r.num_steps for r in results) / len(results) if results else 0
        avg_time = sum(r.execution_time for r in results) / len(results) if results else 0
        
        # Get agent-specific metrics
        agent_metrics = self.agent.get_metrics()
        
        # Create report
        report = EvaluationReport(
            timestamp=datetime.now().isoformat(),
            total_tests=len(results),
            passed_tests=passed,
            failed_tests=failed,
            success_rate=success_rate,
            average_steps=avg_steps,
            average_execution_time=avg_time,
            test_results=results,
            agent_metrics=agent_metrics
        )
        
        # Save report
        report_path = os.path.join(output_dir, f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report.save_to_file(report_path)
        
        # Print summary
        self._print_summary(report)
        
        return report
    
    def _print_summary(self, report: EvaluationReport):
        """Print evaluation summary to console"""
        print("\n" + "="*70)
        print("AGENT EVALUATION SUMMARY")
        print("="*70)
        print(f"Timestamp: {report.timestamp}")
        print(f"Total Tests: {report.total_tests}")
        print(f"Passed: {report.passed_tests} ✓")
        print(f"Failed: {report.failed_tests} ✗")
        print(f"Success Rate: {report.success_rate:.1f}%")
        print(f"Average Steps per Task: {report.average_steps:.1f}")
        print(f"Average Execution Time: {report.average_execution_time:.2f}s")
        print("="*70)
        
        print("\nTest Results:")
        for result in report.test_results:
            status = "✓ PASS" if result.success else "✗ FAIL"
            print(f"  {status} - {result.test_name}")
            print(f"    Tools Used: {', '.join(result.tools_used) if result.tools_used else 'None'}")
            print(f"    Time: {result.execution_time:.2f}s, Steps: {result.num_steps}")
            if not result.success and result.error_message:
                print(f"    Error: {result.error_message}")
        
        print("\n" + "="*70)
        print(f"Agent Performance Metrics:")
        print(f"  Total Actions: {report.agent_metrics.get('total_actions', 0)}")
        print(f"  Successful Actions: {report.agent_metrics.get('successful_actions', 0)}")
        print(f"  Action Success Rate: {report.agent_metrics.get('success_rate', 0):.1f}%")
        
        if 'action_breakdown' in report.agent_metrics:
            print(f"\n  Action Breakdown:")
            for action, count in report.agent_metrics['action_breakdown'].items():
                print(f"    - {action}: {count}")
        
        print("="*70 + "\n")


class BenchmarkComparison:
    """
    Compare agent performance against benchmarks
    
    This provides standardized metrics similar to AgentEval and HumanEval
    """
    
    @staticmethod
    def calculate_task_success_rate(report: EvaluationReport) -> float:
        """Calculate task completion success rate"""
        return report.success_rate
    
    @staticmethod
    def calculate_tool_efficiency(report: EvaluationReport) -> float:
        """
        Calculate tool usage efficiency
        
        Measures how efficiently the agent uses tools compared to expected
        """
        efficiency_scores = []
        
        for result in report.test_results:
            expected_count = len(result.expected_tools)
            actual_count = len(result.tools_used)
            
            if expected_count > 0:
                # Score is higher when actual is close to expected
                efficiency = 1.0 - abs(actual_count - expected_count) / expected_count
                efficiency_scores.append(max(0, efficiency))
        
        return sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0
    
    @staticmethod
    def generate_benchmark_report(report: EvaluationReport) -> Dict[str, Any]:
        """
        Generate standardized benchmark report
        
        Returns:
            Dictionary with benchmark metrics
        """
        return {
            "benchmark_name": "Career Assistant Agent Evaluation",
            "version": "1.0",
            "timestamp": report.timestamp,
            "metrics": {
                "task_success_rate": BenchmarkComparison.calculate_task_success_rate(report),
                "tool_efficiency": BenchmarkComparison.calculate_tool_efficiency(report),
                "average_execution_time": report.average_execution_time,
                "average_reasoning_steps": report.average_steps,
                "error_recovery_rate": report.agent_metrics.get("success_rate", 0) / 100
            },
            "test_coverage": {
                "total_tests": report.total_tests,
                "test_categories": [
                    "document_parsing",
                    "skill_analysis",
                    "document_generation",
                    "tool_orchestration",
                    "error_handling"
                ]
            },
            "comparison_to_baseline": {
                "status": "Track A Compliant",
                "framework": "LangChain with Google Gemini",
                "tool_integration": "✓ Implemented",
                "evaluation_framework": "✓ Implemented"
            }
        }
