import json
import argparse
import requests
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import sys

# Add tora to the Python path if needed
try:
    from tora.client import Tora
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
    from tora.client import Tora


class HyperparameterAdvisor:
    """
    A tool that analyzes training metrics and provides hyperparameter optimization suggestions.
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        api_key: Optional[str] = None,
        llm_endpoint: Optional[str] = None,
        model: str = "gpt-4"
    ):
        """
        Initialize the HyperparameterAdvisor.

        Args:
            experiment_name: Optional name of the experiment to analyze
            api_key: API key for the LLM service
            llm_endpoint: Endpoint URL for the LLM service
            model: Model name to use for analysis
        """
        self.experiment_name = experiment_name
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.llm_endpoint = llm_endpoint or os.environ.get("LLM_ENDPOINT")
        self.model = model

        if not self.api_key:
            print("Warning: No API key provided. Set LLM_API_KEY environment variable or pass api_key.")

        if not self.llm_endpoint:
            print("Warning: No LLM endpoint provided. Set LLM_ENDPOINT environment variable or pass llm_endpoint.")

    def collect_experiment_data(self, experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Collect metrics and hyperparameters from a Tora experiment.

        Args:
            experiment_name: Name of the experiment to analyze. If not provided, uses the instance's experiment_name.

        Returns:
            Dictionary containing experiment data
        """
        exp_name = experiment_name or self.experiment_name
        if not exp_name:
            raise ValueError("Experiment name must be provided")

        try:
            # This assumes that Tora client provides a way to load experiment data
            # This API might need adjustment based on the actual Tora implementation
            client = Tora(name=exp_name, mode="read")
            hyperparams = client.get_hyperparams()
            metrics = client.get_metrics()

            return {
                "experiment_name": exp_name,
                "hyperparameters": hyperparams,
                "metrics": metrics,
                "tags": client.get_tags(),
                "description": client.get_description()
            }
        except Exception as e:
            print(f"Error collecting experiment data: {str(e)}")
            raise

    def analyze_training_trends(self, metrics: Dict[str, List[Tuple[int, float]]]) -> Dict[str, Any]:
        """
        Analyze training metrics to extract trends.

        Args:
            metrics: Dictionary of metrics with name as key and list of (step, value) tuples as values

        Returns:
            Dictionary containing analysis results
        """
        analysis = {}

        # Organize metrics by step
        step_metrics = {}
        for metric_name, values in metrics.items():
            for step, value in values:
                if step not in step_metrics:
                    step_metrics[step] = {}
                step_metrics[step][metric_name] = value

        # Extract final metrics
        if step_metrics:
            max_step = max(step_metrics.keys())
            analysis["final_metrics"] = step_metrics[max_step]

        # Check for overfitting (train accuracy increases while val accuracy decreases)
        train_acc = [v for _, v in metrics.get("train_accuracy", [])]
        val_acc = [v for _, v in metrics.get("val_accuracy", [])]

        if train_acc and val_acc and len(train_acc) == len(val_acc):
            # Simple check: is final train_acc much higher than val_acc?
            if train_acc[-1] - val_acc[-1] > 10:  # >10% difference suggests overfitting
                analysis["overfitting"] = True
            else:
                analysis["overfitting"] = False

        # Check learning rate trends
        lr_values = [v for _, v in metrics.get("learning_rate", [])]
        if lr_values:
            analysis["initial_lr"] = lr_values[0]
            analysis["final_lr"] = lr_values[-1]
            analysis["lr_decay_factor"] = analysis["final_lr"] / analysis["initial_lr"] if analysis["initial_lr"] > 0 else 0

        return analysis

    def generate_prompt(self, experiment_data: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """
        Generate a prompt for the LLM based on the experiment data and analysis.

        Args:
            experiment_data: Dictionary containing experiment data
            analysis: Dictionary containing analysis results

        Returns:
            Prompt string for the LLM
        """
        hyperparams = experiment_data.get("hyperparameters", {})
        metrics = experiment_data.get("metrics", {})

        # Extract final metrics for a more concise prompt
        final_metrics = analysis.get("final_metrics", {})

        prompt = f"""
You are an expert machine learning advisor specializing in hyperparameter optimization.
Analyze the following experiment details and provide suggestions for hyperparameter improvements.

EXPERIMENT: {experiment_data.get('experiment_name')}
DESCRIPTION: {experiment_data.get('description')}
TAGS: {', '.join(experiment_data.get('tags', []))}

CURRENT HYPERPARAMETERS:
{json.dumps(hyperparams, indent=2)}

TRAINING RESULTS:
Final Metrics:
{json.dumps(final_metrics, indent=2)}

ANALYSIS:
"""

        # Add analysis insights
        if analysis.get("overfitting") is not None:
            prompt += f"Overfitting detected: {analysis['overfitting']}\n"

        if "lr_decay_factor" in analysis:
            prompt += f"Learning rate decay factor: {analysis['lr_decay_factor']:.4f}\n"

        prompt += """
Based on the above information, please provide:
1. An assessment of the training results
2. Specific hyperparameter suggestions that could improve performance
3. Reasoning behind each suggestion
4. Any additional techniques that might be beneficial (e.g., regularization, data augmentation)

Format your response as a JSON object with the following keys:
- assessment: A brief assessment of the current training results
- suggestions: An array of suggested hyperparameter changes, each with "param", "current_value", "suggested_value", and "reasoning" fields
- techniques: Additional techniques to consider
"""

        return prompt

    def get_llm_suggestions(self, prompt: str) -> Dict[str, Any]:
        """
        Get hyperparameter suggestions from an LLM.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            Dictionary containing LLM response
        """
        if not self.api_key or not self.llm_endpoint:
            raise ValueError("API key and endpoint must be provided for LLM suggestions")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert machine learning advisor specializing in hyperparameter optimization."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"}
        }

        try:
            response = requests.post(self.llm_endpoint, headers=headers, json=data)
            response.raise_for_status()

            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            return json.loads(content)
        except Exception as e:
            print(f"Error getting LLM suggestions: {str(e)}")
            return {"error": str(e)}

    def run_advisor(self, experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the full hyperparameter advisor pipeline.

        Args:
            experiment_name: Name of the experiment to analyze

        Returns:
            Dictionary containing advisor results
        """
        try:
            # Collect experiment data
            experiment_data = self.collect_experiment_data(experiment_name)

            # Analyze training trends
            analysis = self.analyze_training_trends(experiment_data.get("metrics", {}))

            # Generate prompt
            prompt = self.generate_prompt(experiment_data, analysis)

            # Get LLM suggestions
            suggestions = self.get_llm_suggestions(prompt)

            return {
                "experiment_data": experiment_data,
                "analysis": analysis,
                "suggestions": suggestions
            }
        except Exception as e:
            print(f"Error running advisor: {str(e)}")
            return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Advisor - Get ML model improvement suggestions")
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        required=True,
        help="Name of the experiment to analyze"
    )
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        help="API key for the LLM service (can also use LLM_API_KEY env var)"
    )
    parser.add_argument(
        "--endpoint", "-u",
        type=str,
        help="Endpoint URL for the LLM service (can also use LLM_ENDPOINT env var)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4",
        help="Model name to use for analysis (default: gpt-4)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file to save suggestions (default: print to stdout)"
    )

    args = parser.parse_args()

    advisor = HyperparameterAdvisor(
        experiment_name=args.experiment,
        api_key=args.api_key,
        llm_endpoint=args.endpoint,
        model=args.model
    )

    results = advisor.run_advisor()

    if "error" in results:
        print(f"Error: {results['error']}")
        sys.exit(1)

    suggestions = results.get("suggestions", {})

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(suggestions, f, indent=2)
        print(f"Suggestions saved to {args.output}")
    else:
        print(json.dumps(suggestions, indent=2))


if __name__ == "__main__":
    main()
