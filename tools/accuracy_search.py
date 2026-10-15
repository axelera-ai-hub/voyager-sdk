#!/usr/bin/env python
# Copyright Axelera AI, 2025
# Accuracy Search Script for Specified Model-Zoo Models

import argparse
import csv
import datetime
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import yaml

try:
    import pandas as pd
except ModuleNotFoundError:
    import subprocess

    subprocess.check_call(["pip", "install", "pandas"])
    import pandas as pd  # Import it again after installation

    print("Pandas installed successfully!")


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def run_command(command, shell=True, verbose=False):
    """Executes a shell command with optional verbose output and error handling."""
    try:
        if verbose:
            print(f"{Colors.OKGREEN}--- Running Command: {command} ---{Colors.ENDC}")
            process = subprocess.Popen(
                command,
                shell=shell,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=0,
            )
            for line in iter(process.stdout.readline, ""):
                print(line.strip())
                sys.stdout.flush()
            returncode = process.wait()
            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, command)
            return ""
        else:
            result = subprocess.run(
                command, shell=shell, capture_output=True, text=True, check=True
            )
            return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"{Colors.FAIL}Error running command: {command}{Colors.ENDC}")
        print(f"  Return code: {e.returncode}")
        if not verbose:
            print(f"  Stdout: {e.stdout}")
            print(f"  Stderr: {e.stderr}")
        raise


def get_applicable_metric(output):
    """Extracts a key metric value from a string."""
    custom_pattern = re.compile(r"Key Metric \(([^)]+)\):\s*([\d.]+)%?")
    match = custom_pattern.search(output)
    if match:
        try:
            metric_name = match.group(1)
            metric_value = float(match.group(2))
            return (metric_name, metric_value)
        except (ValueError, IndexError):
            return (None, None)
    return (None, None)


def calculate_drop(accuracy, target_accuracy):
    """Calculates absolute and relative accuracy drop."""
    if target_accuracy is None:
        return None, None
    drop = target_accuracy - accuracy
    rel_drop = (drop / target_accuracy) * 100 if target_accuracy != 0 else None
    return drop, rel_drop


def format_accuracy_output(
    metric_name, accuracy, target_accuracy, best_accuracy_so_far=None, best_rel_drop_so_far=None
):
    """Formats the accuracy output string."""
    drop, rel_drop = calculate_drop(accuracy, target_accuracy)
    output = f"{Colors.OKBLUE}{metric_name}: {accuracy:.2f}{Colors.ENDC}"
    if target_accuracy is not None:
        output += f"  Target {target_accuracy:.2f}, Drop={drop:.2f}, Rel Drop={rel_drop:.2f}%"
    if best_accuracy_so_far is not None:
        output += f"  (best set: {best_accuracy_so_far:.2f}, {best_rel_drop_so_far:.2f}%)"
    return output


def log_results(
    log_file, model_name, num_cal_images, seed, attempt, accuracy, target_accuracy, zip_file=None
):
    """Logs the results of a single run to a CSV file."""
    drop, rel_drop = calculate_drop(accuracy, target_accuracy)
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as csvfile:
        fieldnames = [
            'timestamp',
            'model_name',
            'num_cal_images',
            'seed',
            'attempt',
            'accuracy',
            'target_accuracy',
            'drop',
            'rel_drop',
            'zip_file',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                'model_name': model_name,
                'num_cal_images': num_cal_images,
                'seed': seed,
                'attempt': attempt,
                'accuracy': accuracy,
                'target_accuracy': target_accuracy if target_accuracy is not None else "N/A",
                'drop': drop if drop is not None else "N/A",
                'rel_drop': rel_drop if rel_drop is not None else "N/A",
                'zip_file': zip_file if zip_file is not None else "N/A",
            }
        )


def get_target_accuracy(model_name, verbose):
    """Gets the target (FP32) accuracy for a given model."""
    print(f"{Colors.OKCYAN}Determining target accuracy for {model_name}...{Colors.ENDC}")
    inference_command = f"python3 inference.py mc-{model_name} dataset --pipe=torch --no-display"
    output = run_command(inference_command, verbose=verbose)
    metric_name, accuracy = get_applicable_metric(output)
    if accuracy is None:
        print(
            f"{Colors.FAIL}Error: Could not determine target accuracy for {model_name}.{Colors.ENDC}"
        )
        return None
    print(f"{Colors.OKGREEN}Target accuracy for {model_name}: {accuracy:.2f}{Colors.ENDC}")
    return accuracy


def search_best_accuracy(
    model_name,
    target_accuracy,
    target_rel_drop,
    num_cal_images_list,
    cal_seeds,
    n_try,
    use_representative_images,
    verbose,
    log_file,
):
    """Searches for the best accuracy, logs results, and handles early termination."""
    best_accuracy = -1.0
    worst_accuracy = float('inf')  # Initialize worst accuracy
    best_config = {}
    best_rel_drop_so_far = float('inf')

    if n_try > 1:
        print(
            f"{Colors.WARNING}Warning: n_try > 1.  Running in deterministic mode with num_cal_images={num_cal_images_list[0]} and seed={cal_seeds[0]}{Colors.ENDC}"
        )
        num_cal_images_list = [num_cal_images_list[0]]
        cal_seeds = [cal_seeds[0]]

    for num_cal_images in num_cal_images_list:
        for seed in cal_seeds:
            for attempt in range(1, n_try + 1):
                print(
                    f"{Colors.HEADER}Running for {model_name}: num_cal_images={num_cal_images}, seed={seed}, attempt={attempt}{Colors.ENDC}"
                )
                run_command(f"make NN=mc-{model_name} clean", verbose=verbose)
                use_representative_images_flag = (
                    "--default-representative-images"
                    if use_representative_images
                    else "--no-default-representative-images"
                )
                deploy_command = f"python3 deploy.py mc-{model_name} --export --num-cal-images={num_cal_images} --cal-seed={seed} {use_representative_images_flag}"
                try:
                    run_command(deploy_command, verbose=verbose)
                except subprocess.CalledProcessError:
                    print(
                        f"{Colors.WARNING}Warning: Deployment failed for num_cal_images={num_cal_images}. Skipping this and larger values.{Colors.ENDC}"
                    )
                    break  # Skip to the next seed
                inference_command = (
                    f"python3 inference.py mc-{model_name} dataset --pipe=torch-aipu --no-display"
                )
                output = run_command(inference_command, verbose=verbose)

                metric_name, accuracy = get_applicable_metric(output)
                if accuracy is None:
                    print(f"{Colors.WARNING}Warning: Could not find accuracy.{Colors.ENDC}")
                    log_results(
                        log_file, model_name, num_cal_images, seed, attempt, "N/A", target_accuracy
                    )
                    continue

                drop, rel_drop = calculate_drop(accuracy, target_accuracy)

                # Update worst accuracy
                worst_accuracy = min(worst_accuracy, accuracy)

                if rel_drop is not None and rel_drop < best_rel_drop_so_far:
                    best_rel_drop_so_far = rel_drop
                    best_accuracy_so_far = accuracy
                elif rel_drop is None and accuracy > best_accuracy:
                    best_accuracy_so_far = accuracy
                    best_rel_drop_so_far = None
                else:
                    best_accuracy_so_far = best_accuracy

                print(
                    format_accuracy_output(
                        metric_name,
                        accuracy,
                        target_accuracy,
                        best_accuracy_so_far,
                        best_rel_drop_so_far,
                    )
                )

                original_zip = os.path.join("exported", f"{model_name}.zip")
                attempt_suffix = f"_{attempt}" if n_try > 1 else ""
                new_zip_name = (
                    f"{model_name}_{accuracy}_{num_cal_images}_{seed}{attempt_suffix}.zip"
                )
                new_zip = os.path.join("exported", new_zip_name)

                # Log results *before* checking for best accuracy and moving ZIP
                log_results(
                    log_file,
                    model_name,
                    num_cal_images,
                    seed,
                    attempt,
                    accuracy,
                    target_accuracy,
                    new_zip_name if accuracy > best_accuracy else None,
                )

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_config = {
                        "best_accuracy": best_accuracy,
                        "num_cal_images": num_cal_images,
                        "seed": seed,
                        "attempt": attempt,
                        "zip_file": new_zip_name,
                        "rel_drop": rel_drop if rel_drop is not None else "N/A",
                    }
                    if os.path.exists(original_zip):
                        shutil.move(original_zip, new_zip)
                    else:
                        print(
                            f"{Colors.WARNING}Warning: Zip not found: {original_zip}{Colors.ENDC}"
                        )

                if target_rel_drop is not None and rel_drop <= target_rel_drop:
                    print(f"{Colors.OKGREEN}Early termination: Target drop met.{Colors.ENDC}")
                    best_config["worst_accuracy"] = worst_accuracy
                    return best_config
            else:
                continue  # Only executed if the inner loop did NOT break
            break  # Only executed if the inner loop DID break

    best_config["worst_accuracy"] = worst_accuracy
    print(f"\n{Colors.OKCYAN}Best configuration for {model_name}:{Colors.ENDC}")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    return best_config


def generate_summary(log_file, models=None, single_model_target_accuracy=None):
    """Generates a summary with statistics from the log file."""
    try:
        df = pd.read_csv(log_file)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"{Colors.FAIL}Error: Invalid log file: {log_file}{Colors.ENDC}")
        return None

    summary = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "source_log_file": log_file,
        "results": {},
    }

    for model_name in df['model_name'].unique():
        model_df = df[df['model_name'] == model_name]
        accuracies = pd.to_numeric(model_df['accuracy'], errors='coerce').dropna()
        rel_drops = pd.to_numeric(model_df['rel_drop'], errors='coerce').dropna()
        best_row = model_df.loc[model_df['accuracy'].idxmax()]

        if models is not None:
            target_accuracy = (
                models[model_name][0]
                if isinstance(models[model_name], tuple)
                else models[model_name]
            )
        else:
            target_accuracy = single_model_target_accuracy

        best_accuracy_drop, best_rel_drop = calculate_drop(best_row['accuracy'], target_accuracy)
        worst_accuracy = model_df['accuracy'].min()

        overall_mean_accuracy = accuracies.mean()
        overall_std_accuracy = accuracies.std()
        overall_mean_rel_drop = rel_drops.mean()
        overall_std_rel_drop = rel_drops.std()

        # Create a DataFrame for the per-factor statistics
        factor_data = []
        for factor in ['num_cal_images', 'seed']:
            grouped = model_df.groupby(factor)
            for name, group in grouped:
                factor_data.append(
                    {
                        "factor": factor,
                        "factor_value": name,
                        "mean_accuracy": group['accuracy'].mean(),
                        "std_accuracy": group['accuracy'].std(),
                        "mean_rel_drop": group['rel_drop'].mean(),
                        "std_rel_drop": group['rel_drop'].std(),
                        "num_samples": len(group),
                    }
                )
        factor_df = pd.DataFrame(factor_data)

        # Create a dictionary for the overall model summary
        model_summary = {
            "best_accuracy": float(best_row['accuracy']),
            "worst_accuracy": float(worst_accuracy),
            "best_accuracy_drop": (
                float(best_accuracy_drop) if best_accuracy_drop is not None else "N/A"
            ),
            "best_rel_drop": float(best_rel_drop) if best_rel_drop is not None else "N/A",
            "target_accuracy": float(target_accuracy) if target_accuracy is not None else "N/A",
            "num_cal_images": int(best_row['num_cal_images']),
            "seed": int(best_row['seed']),
            "attempt": int(best_row['attempt']),
            "zip_file": best_row['zip_file'] if pd.notna(best_row['zip_file']) else "N/A",
            "mean_accuracy": float(overall_mean_accuracy),
            "std_accuracy": float(overall_std_accuracy),
            "mean_rel_drop": float(overall_mean_rel_drop),
            "std_rel_drop": float(overall_std_rel_drop),
            "factor_statistics": factor_df.to_dict(
                orient='records'
            ),  # Convert DataFrame to list of dicts
        }

        summary["results"][model_name] = model_summary

    return summary


def draw_accuracy_plots(log_file, output_dir, models=None, single_model_target_accuracy=None):
    """Generates plots, saving them to the specified output directory."""
    import matplotlib
    import matplotlib.pyplot as plt

    try:
        df = pd.read_csv(log_file)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"{Colors.FAIL}Error: Invalid log file: {log_file}{Colors.ENDC}")
        return

    for model_name in df['model_name'].unique():
        model_df = df[df['model_name'] == model_name]
        accuracies = pd.to_numeric(model_df['accuracy'], errors='coerce').dropna()
        if accuracies.empty:
            print(
                f"{Colors.WARNING}Warning: No valid accuracy data for {model_name}. Skipping plots.{Colors.ENDC}"
            )
            continue

        # Determine target accuracy
        if models is not None:
            target_accuracy = (
                models[model_name][0]
                if isinstance(models[model_name], tuple)
                else models[model_name]
            )
        else:
            target_accuracy = single_model_target_accuracy

        # --- Plot 1: Accuracy vs. Number of Calibration Images (Improved) ---
        plt.figure(figsize=(12, 7))
        best_acc = -1  # Keep track of best accuracy for highlighting
        for seed in model_df['seed'].unique():
            seed_df = model_df[model_df['seed'] == seed].sort_values('num_cal_images')
            # Calculate a rolling average for smoother lines
            rolling_avg = seed_df['accuracy'].rolling(window=max(1, len(seed_df) // 5)).mean()
            plt.plot(
                seed_df['num_cal_images'], rolling_avg, marker='o', label=f'Seed {seed}', alpha=0.7
            )
            # Find and highlight the best accuracy
            for _, row in seed_df.iterrows():
                if row['accuracy'] > best_acc:
                    best_acc = row['accuracy']
                    best_cal_images = row['num_cal_images']
                    best_seed = row['seed']

        # Highlight the overall best accuracy
        plt.plot(
            best_cal_images,
            best_acc,
            marker='*',
            color='red',
            markersize=12,
            label=f'Best: {best_acc:.2f}',
        )

        # Add target accuracy line
        if target_accuracy is not None:
            plt.axhline(
                y=target_accuracy,
                color='gray',
                linestyle='--',
                label=f'Target: {target_accuracy:.2f}',
            )

        # Dynamic y-axis limits
        ymin = min(
            accuracies.min() * 0.98,
            target_accuracy * 0.98 if target_accuracy else accuracies.min() * 0.98,
        )
        ymax = max(
            accuracies.max() * 1.02,
            target_accuracy * 1.02 if target_accuracy else accuracies.max() * 1.02,
        )
        plt.ylim(ymin, ymax)

        plt.xlabel('Number of Calibration Images')
        plt.ylabel('Accuracy (Rolling Avg)')
        plt.title(f'Accuracy vs. Calibration Images for {model_name} (Smoothed)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_accuracy_vs_cal_images.png'))
        plt.close()

        # --- Plot 2: Accuracy vs. Seed (Improved) ---
        plt.figure(figsize=(12, 7))
        best_acc = -1
        for num_cal_images in model_df['num_cal_images'].unique():
            cal_df = model_df[model_df['num_cal_images'] == num_cal_images].sort_values('seed')
            # Calculate rolling average
            rolling_avg = cal_df['accuracy'].rolling(window=max(1, len(cal_df) // 5)).mean()
            plt.plot(
                cal_df['seed'],
                rolling_avg,
                marker='o',
                label=f'Cal Images {num_cal_images}',
                alpha=0.7,
            )
            for _, row in cal_df.iterrows():
                if row['accuracy'] > best_acc:
                    best_acc = row['accuracy']
                    best_cal_images = row['num_cal_images']
                    best_seed = row['seed']
        plt.plot(
            best_seed,
            best_acc,
            marker='*',
            color='red',
            markersize=12,
            label=f'Best: {best_acc:.2f}',
        )

        # Add target accuracy line
        if target_accuracy is not None:
            plt.axhline(
                y=target_accuracy,
                color='gray',
                linestyle='--',
                label=f'Target: {target_accuracy:.2f}',
            )

        # Dynamic y-axis limits
        ymin = min(
            accuracies.min() * 0.98,
            target_accuracy * 0.98 if target_accuracy else accuracies.min() * 0.98,
        )
        ymax = max(
            accuracies.max() * 1.02,
            target_accuracy * 1.02 if target_accuracy else accuracies.max() * 1.02,
        )
        plt.ylim(ymin, ymax)

        plt.xlabel('Seed')
        plt.ylabel('Accuracy (Rolling Avg)')
        plt.title(f'Accuracy vs. Seed for {model_name} (Smoothed)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_accuracy_vs_seed.png'))
        plt.close()

        # --- Plot 3: Enhanced Box Plot with Variance (Combined) ---
        plt.figure(figsize=(14, 8))

        # Create a pivot table for the boxplot
        pivot_df = model_df.pivot_table(index='num_cal_images', columns='seed', values='accuracy')

        # Boxplot:  Get the Axes object *after* plotting
        ax = plt.gca()  # Get current axes
        pivot_df.boxplot(ax=ax, patch_artist=True)  # Plot onto the axes

        # Access boxplot elements through the Axes object
        boxes = [artist for artist in ax.artists]

        # Customize box colors
        colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow', 'lightcyan']
        # Ensure we don't try to apply more colors than there are boxes
        num_boxes = len(boxes)
        colors = colors[:num_boxes]

        for patch, color in zip(boxes, colors):  # Access boxes directly
            patch.set_facecolor(color)

        # Add individual data points (jitter for clarity)
        for i, num_cal_images in enumerate(pivot_df.index):
            y = pivot_df.loc[num_cal_images].values
            x = np.random.normal(i + 1, 0.04, size=len(y))  # Add jitter
            plt.plot(x, y, 'r.', alpha=0.4)

        plt.xlabel('Number of Calibration Images (Groups) / Seed (Within Group)')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Distribution for {model_name} (Std Dev: {accuracies.std():.2f})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_accuracy_boxplot.png'))
        plt.close()

        # --- Plot 4: Std Dev of Accuracy vs. Num Cal Images ---
        plt.figure(figsize=(10, 6))
        std_by_cal = model_df.groupby('num_cal_images')['accuracy'].std()
        plt.plot(
            std_by_cal.index, std_by_cal.values, marker='o', linestyle='-', color='purple'
        )  # Added linestyle
        plt.xlabel('Number of Calibration Images')
        plt.ylabel('Standard Deviation of Accuracy')
        plt.title(f'Stability (Std Dev) vs. Calibration Images for {model_name}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_std_vs_cal_images.png'))
        plt.close()

    # --- Plot 5: Cross-Model Relative Drop Variance (Improved) ---
    plt.figure(figsize=(14, 8))
    rel_drops_by_model = {}
    target_drops = {}

    for model in df['model_name'].unique():
        rel_drops = pd.to_numeric(
            df[df['model_name'] == model]['rel_drop'], errors='coerce'
        ).dropna()
        if not rel_drops.empty:  # Only add if data exists
            rel_drops_by_model[model] = rel_drops
        # Get target drops if available
        if models and model in models:
            if isinstance(models[model], tuple):
                target_drops[model] = (
                    models[model][1] * 100 if models[model][1] is not None else None
                )  # convert to percentage

    if rel_drops_by_model:  # Check if any data was collected
        # Sort models by median relative drop
        sorted_models = sorted(rel_drops_by_model.items(), key=lambda item: np.median(item[1]))
        sorted_model_names = [model for model, _ in sorted_models]
        sorted_rel_drops = [rel_drops_by_model[model] for model in sorted_model_names]

        # Use plt.gca() to get the current Axes object
        ax = plt.gca()
        ax.boxplot(sorted_rel_drops, labels=sorted_model_names, patch_artist=True)

        # Access boxplot elements through the Axes object
        boxes = [artist for artist in ax.artists]

        colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow', 'lightcyan']
        num_boxes = len(boxes)
        colors = colors[:num_boxes]
        for patch, color in zip(boxes, colors):
            patch.set_facecolor(color)

        # Add target drop line (if applicable)
        if target_drops:
            avg_target_drop = np.mean([drop for drop in target_drops.values() if drop is not None])
            if not np.isnan(avg_target_drop):  # Check if we have a valid average
                plt.axhline(
                    y=avg_target_drop,
                    color='gray',
                    linestyle='--',
                    label=f'Avg Target Drop: {avg_target_drop:.2f}%',
                )
                plt.legend()

        plt.xlabel('Model')
        plt.ylabel('Relative Accuracy Drop (%)')
        plt.title('Relative Accuracy Drop Distribution Across Models')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cross_model_rel_drop_variance.png'))
        plt.close()
    else:
        print(
            f"{Colors.WARNING}Warning: No valid relative drop data for any model. Skipping cross-model plot.{Colors.ENDC}"
        )


def parse_models_yaml(yaml_string):
    """
    Parses a YAML string in the format 'filepath:section' and returns a list of models.

    Args:
        yaml_string: A string in the format 'filepath:section'.

    Returns:
        A list of model names, or None if an error occurs.
    """
    try:
        filepath, section = yaml_string.split(":")
        filepath = filepath.strip()  # Remove leading/trailing whitespace
        section = section.strip()

        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        if section in data:
            return data[section]
        else:
            print(
                f"{Colors.FAIL}Error: Section '{section}' not found in '{filepath}'.{Colors.ENDC}"
            )
            return None

    except FileNotFoundError:
        print(f"{Colors.FAIL}Error: File not found: '{filepath}'.{Colors.ENDC}")
        return None
    except ValueError:
        print(
            f"{Colors.FAIL}Error: Invalid format for --models. Use 'filepath:section'.{Colors.ENDC}"
        )
        return None
    except (yaml.YAMLError, KeyError) as e:
        print(f"{Colors.FAIL}Error parsing YAML file: {e}{Colors.ENDC}")
        return None
    except Exception as e:
        print(f"{Colors.FAIL}An unexpected error occurred: {e}{Colors.ENDC}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Accuracy Search Script for YOLO Models")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "-p",
        "--plot",
        nargs='?',
        const=True,
        default=False,
        help="Generate plots. If a filename is provided, plots are generated from that file. "
        "If no filename is provided, the accuracy search is run, and plots are generated.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--model",
        type=str,
        default=None,
        help="Run for a single model. Overrides the 'models' dictionary.",
    )
    group.add_argument(
        "--models",
        type=str,
        default=None,
        help="Path to YAML file and section containing the list of models (e.g., 'models.yaml:VALIDATION').",
    )
    parser.add_argument(
        "--n_try",
        type=int,
        default=1,
        help="Number of attempts per configuration (default: 1).  >1 for deterministic validation.",
    )
    parser.add_argument(
        "--use-representative-images",
        action="store_true",
        default=False,
        help="Use the default representative images for search.",
    )

    args = parser.parse_args()

    models = {
        "yolov8m-coco-onnx": (50.16, 0.01621),
        "yolov8l-coco-onnx": (52.83, 0.01261),
        "yolo11n-coco-onnx": (39.17, 0.00481),
        "yolo11s-coco-onnx": (46.54, 0.00391),
        "yolo11m-coco-onnx": (51.31, 0.0031),
        "yolo11l-coco-onnx": (53.23, 0.00391),
        "yolov8npose-coco-onnx": (51.11, 0.01811),
        "yolov8spose-coco-onnx": (60.56, 0.0241),
        "yolo11npose-coco-onnx": (51.15, 0.0151),
        "yolo11nseg-coco-onnx": (70.50, 0.0151),
        "yolov8s-coco-onnx": (44.8, 0.01441),
        "yolo11lpose-coco-onnx": (67.44, 0.0151),
        "yolo11npose-coco-onnx": (51.15, 0.0151),
        "yolo11lseg-coco-onnx": (71.76, 0.0151),
        "yolo11nseg-coco-onnx": (56.49, 0.0131),
        "yolov8lpose-coco-onnx": (68.39, 0.0151),
        "yolov8lseg-coco-onnx": (70.50, 0.0151),
        "yolov8nseg-coco-onnx": (54.12, 0.01401),
        "yolov8sseg-coco-onnx": (63.13, 0.01011),
    }

    if args.models:
        model_names = parse_models_yaml(args.models)
        if model_names is None:
            sys.exit(1)  # Exit if parsing failed
        models = {}
        for model_name in model_names:
            target_accuracy = get_target_accuracy(model_name, args.verbose)
            if target_accuracy is None:
                print(
                    f"{Colors.WARNING}Warning: Could not retrieve target accuracy for {model_name}. Skipping.{Colors.ENDC}"
                )
                continue  # Skip this model and move to the next
            models[model_name] = (target_accuracy, None)  # Store as (accuracy, rel_drop)
        if not models:
            print(f"{Colors.FAIL}Error: No valid models found in YAML file. Exiting.{Colors.ENDC}")
            sys.exit(1)
    elif args.model:  # Single model
        target_accuracy = get_target_accuracy(args.model, args.verbose)
        if target_accuracy is None:
            print(
                f"{Colors.FAIL}Error: Could not retrieve target accuracy for {args.model}. Exiting.{Colors.ENDC}"
            )
            sys.exit(1)
        models = {args.model: (target_accuracy, None)}
    else:
        target_accuracy = None  # set to None when it is not single model case

    num_cal_images_list = [100, 200, 400]
    cal_seeds = [19, 24, 41, 129, 141]

    print(
        f"{Colors.OKGREEN}Running for {len(models)} models with {len(num_cal_images_list)} cal images and {len(cal_seeds)} seeds{Colors.ENDC}"
    )

    # Create results directory
    if not os.path.exists("results"):
        os.makedirs("results", exist_ok=True)  # Add exist_ok=True here as well
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join("results", timestamp)
    os.makedirs(output_dir, exist_ok=True)  # Add exist_ok=True here

    if args.plot is True or (args.plot is False and not args.model):
        log_file = os.path.join(output_dir, f"accuracy_log.csv")  # Put log in results dir
        all_best_configs = {}
        for model_name, model_info in models.items():
            if isinstance(model_info, tuple):
                target_accuracy, target_rel_drop = model_info
            else:
                target_accuracy, target_rel_drop = model_info, None  # No target rel drop

            best_config = search_best_accuracy(
                model_name,
                target_accuracy,
                target_rel_drop,
                num_cal_images_list,
                cal_seeds,
                args.n_try,
                args.use_representative_images,
                args.verbose,
                log_file,
            )
            if best_config:
                all_best_configs[model_name] = best_config

        summary = generate_summary(
            log_file, models if not args.model else None, target_accuracy if args.model else None
        )
        summary_file = os.path.join(
            output_dir, f"accuracy_summary.yaml"
        )  # Put summary in results dir
        with open(summary_file, "w") as f:
            yaml.dump(summary, f, indent=4)
        print(f"{Colors.OKGREEN}Summary written to: {summary_file}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}Results logged to: {log_file}{Colors.ENDC}")
        if args.plot:
            draw_accuracy_plots(
                log_file,
                output_dir,
                models if not args.model else None,
                target_accuracy if args.model else None,
            )

    elif isinstance(args.plot, str):  # Plot from existing file
        log_file = args.plot
        # Create a results directory even when plotting from an existing file
        if not os.path.exists("results"):
            os.makedirs("results", exist_ok=True)  # Add exist_ok=True
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_dir = os.path.join("results", timestamp)
        os.makedirs(output_dir, exist_ok=True)  # Add exist_ok=True

        summary = generate_summary(
            log_file, models if not args.model else None, target_accuracy if args.model else None
        )
        if summary:
            summary_file = os.path.join(
                output_dir, f"accuracy_summary.yaml"
            )  # Put summary in results dir
            with open(summary_file, "w") as f:
                yaml.dump(summary, f, indent=4)
            print(f"{Colors.OKGREEN}Summary written to: {summary_file}{Colors.ENDC}")
        draw_accuracy_plots(
            log_file,
            output_dir,
            models if not args.model else None,
            target_accuracy if args.model else None,
        )


if __name__ == "__main__":
    main()
