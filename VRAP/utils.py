#!/usr/bin/env python3
"""
Utility functions for OS Agent
Common helper functions used across the project
"""

import json
import os
import datetime
import random
import re


def clean_ansi_sequences(text):
    """Clean ANSI escape sequences from text to make it terminal-display consistent
    
    Args:
        text: Raw text that may contain ANSI escape sequences
        
    Returns:
        str: Clean text without ANSI escape sequences
    """
    if not text:
        return text
    
    # Remove ANSI escape sequences
    # CSI sequences: ESC [ ... m (colors, formatting)
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    
    # OSC sequences: ESC ] ... BEL or ESC ] ... ESC \
    text = re.sub(r'\x1b\][^\x07\x1b]*[\x07\x1b\\]', '', text)
    
    # Mode switching sequences: ESC [ ? ... h/l
    text = re.sub(r'\x1b\[\?[0-9;]*[hl]', '', text)
    
    # Other common escape sequences
    text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)
    text = re.sub(r'\x1b\([AB]', '', text)
    text = re.sub(r'\x1b[=>]', '', text)
    
    # Remove other control characters but keep newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)
    
    return text

def read_os_dataset(path):
    """Read dataset from JSON or JSONL file"""
    print(f"Reading dataset from: {path}")
    if path.endswith(".json"):
        with open(path, "r") as file:
            return json.load(file)
    elif path.endswith(".jsonl"):
        data = []
        with open(path, "r") as file:
            for line in file:
                data.append(json.loads(line))
        return data


def save_result_to_file(item, verification_success, output_dir, execution_metadata=None, input_filename=None):
    """Save verification result to framework-specific directory with JSR field and execution metadata
    
    Args:
        item: Original dataset item with all fields
        verification_success: Boolean indicating if verification was successful
        output_dir: Output directory path
        execution_metadata: Dict containing agent_framework, provider, and model info
        input_filename: Path to input dataset file (used for filename prefix)
    """
    try:
        # Determine framework-specific directory
        framework_name = execution_metadata.get("agent_framework", "Unknown") if execution_metadata else "Unknown"
        
        # Map framework names to directory names
        framework_dir_map = {
            "ReactOSAgent": "ReactOSAgent",
            "MITRE_ATTCK_OSAgent": "MITRE_ATTCK_OSAgent", 
            "AutoGPT_OSAgent": "AutoGPT"
        }
        
        framework_dir = framework_dir_map.get(framework_name, framework_name)
        
        # Create framework-specific result directory
        result_dir = os.path.join(os.path.abspath("results"), framework_dir)
        os.makedirs(result_dir, exist_ok=True)
        
        # Generate filename with input file prefix
        technique_ids = item.get("technique_id", ["unknown"])
        technique_id = "_".join(technique_ids) if isinstance(technique_ids, list) and technique_ids else "unknown"
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = random.randint(1000, 9999)
        
        if input_filename:
            # Extract filename without extension from input path
            input_basename = os.path.splitext(os.path.basename(input_filename))[0]
            filename = f"{input_basename}_{technique_id}_{timestamp}_{random_suffix}_result.json"
        else:
            filename = f"{technique_id}_{timestamp}_{random_suffix}_result.json"
        
        # Save file with all original fields plus JSR field and execution metadata
        result_file_path = os.path.join(result_dir, filename)
        # Preserve all original fields and add JSR (Jailbreak Success Result) and execution metadata
        result_data = {**item, "JSR": verification_success}
        
        # Add execution metadata if provided
        if execution_metadata:
            result_data.update(execution_metadata)
        
        with open(result_file_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        jsr_status = "True" if verification_success else "False"
        print(f"Result saved with JSR={jsr_status}: {result_file_path}")
        return result_file_path
        
    except Exception as e:
        print(f"Save result error: {e}")
        return None


def save_batch_results_to_file(results_list, execution_metadata=None, input_filename=None):
    """Save batch results to a single file with custom naming format
    
    Args:
        results_list: List of (item, verification_success) tuples
        execution_metadata: Dict containing agent_framework, provider, and model info
        input_filename: Path to input dataset file (used for filename prefix)
        
    Returns:
        str: Path to saved file
    """
    try:
        if not results_list:
            print("No results to save")
            return None
            
        # Determine framework-specific directory
        framework_name = execution_metadata.get("agent_framework", "Unknown") if execution_metadata else "Unknown"
        
        # Map framework names to directory names
        framework_dir_map = {
            "ReactOSAgent": "ReactOSAgent",
            "MITRE_ATTCK_OSAgent": "MITRE_ATTCK_OSAgent", 
            "AutoGPT_OSAgent": "AutoGPT"
        }
        
        framework_dir = framework_dir_map.get(framework_name, framework_name)
        
        # Create framework-specific result directory
        result_dir = os.path.join("results", framework_dir)
        os.makedirs(result_dir, exist_ok=True)
        
        # Extract model name
        model_name = execution_metadata.get("model", "unknown") if execution_metadata else "unknown"
        
        # Get first and last item IDs
        first_item = results_list[0][0]
        last_item = results_list[-1][0]
        
        # Extract unique_id or id or use index
        start_id = first_item.get("unique_id") or first_item.get("id", "0")
        end_id = last_item.get("unique_id") or last_item.get("id", str(len(results_list)-1))
        
        # Convert to string if needed
        start_id = str(start_id)
        end_id = str(end_id)
        
        # Calculate success rate
        successful_count = sum(1 for _, success in results_list if success)
        total_count = len(results_list)
        success_rate = int((successful_count / total_count) * 100) if total_count > 0 else 0
        
        # Generate timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate filename with input file prefix: [输入文件名无后缀]_模型_起始id_结束id_成功率百分比*100保留整数_时间戳.json
        if input_filename:
            # Extract filename without extension from input path
            input_basename = os.path.splitext(os.path.basename(input_filename))[0]
            filename = f"{input_basename}_{model_name}_{start_id}_{end_id}_{success_rate}_{timestamp}.json"
        else:
            filename = f"{model_name}_{start_id}_{end_id}_{success_rate}_{timestamp}.json"
        
        # Prepare batch data
        batch_data = {
            "batch_info": {
                "model": model_name,
                "start_id": start_id,
                "end_id": end_id,
                "success_rate_percent": success_rate,
                "successful_count": successful_count,
                "total_count": total_count,
                "timestamp": timestamp
            },
            "execution_metadata": execution_metadata or {},
            "results": []
        }
        
        # Add all results
        for item, verification_success in results_list:
            result_data = {**item, "JSR": verification_success}
            batch_data["results"].append(result_data)
        
        # Save file
        result_file_path = os.path.join(result_dir, filename)
        with open(result_file_path, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, indent=2, ensure_ascii=False)
        
        print(f"Batch results saved: {result_file_path}")
        print(f"Success rate: {successful_count}/{total_count} ({success_rate}%)")
        return result_file_path
        
    except Exception as e:
        print(f"Save batch results error: {e}")
        return None


# Global variables to track experiment session
_experiment_session = None
_terminal_log = []
_full_terminal_output = []
_realtime_summary = {}  # Track realtime summary data

def init_experiment_session(results_list, execution_metadata, input_filename, start_timestamp):
    """Initialize experiment session with final naming format
    
    Args:
        results_list: List of (item, verification_success) tuples (can be empty initially)
        execution_metadata: Dict containing agent_framework, provider, and model info
        input_filename: Path to input dataset file
        start_timestamp: Timestamp when experiment started
        
    Returns:
        dict: Session info with experiment folder path and final filename
    """
    global _experiment_session, _terminal_log
    
    try:
        # Determine framework-specific directory
        framework_name = execution_metadata.get("agent_framework", "Unknown") if execution_metadata else "Unknown"
        framework_dir_map = {
            "ReactOSAgent": "ReactOSAgent",
            "MITRE_ATTCK_OSAgent": "MITRE_ATTCK_OSAgent", 
            "AutoGPT_OSAgent": "AutoGPT"
        }
        framework_dir = framework_dir_map.get(framework_name, framework_name)
        
        # Extract model name and sanitize it for filesystem
        model_name = execution_metadata.get("model", "unknown") if execution_metadata else "unknown"
        # Replace slashes and other problematic characters with hyphens
        sanitized_model_name = model_name.replace("/", "-").replace("\\", "-").replace(":", "-")
        
        # Extract input filename without extension
        input_basename = "unknown"
        if input_filename:
            input_basename = os.path.splitext(os.path.basename(input_filename))[0]
        
        # Generate experiment folder name: 输入文件名_模型名_时间戳
        experiment_name = f"{input_basename}_{sanitized_model_name}_{start_timestamp}"
        
        # Create experiment directory
        result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", framework_dir)
        experiment_dir = os.path.join(result_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        print("===============>experiment_dir: ", experiment_dir)
        
        # Create records subdirectory for detailed attempt logs
        records_dir = os.path.join(experiment_dir, "records")
        os.makedirs(records_dir, exist_ok=True)
        
        # Initialize session
        _experiment_session = {
            "experiment_name": experiment_name,
            "experiment_dir": experiment_dir,
            "records_dir": records_dir,
            "result_dir": result_dir,
            "model_name": model_name,
            "sanitized_model_name": sanitized_model_name,
            "start_timestamp": start_timestamp,
            "execution_metadata": execution_metadata,
            "input_filename": input_filename
        }
        
        # Initialize global variables
        _terminal_log = []
        _realtime_summary.clear()  # Clear any previous data
        log_terminal_interaction(f"实验开始: {experiment_name}")
        log_terminal_interaction(f"使用模型: {model_name}")
        log_terminal_interaction(f"框架: {framework_name}")
        
        return _experiment_session
        
    except Exception as e:
        print(f"Initialize experiment session error: {e}")
        return None

def log_terminal_interaction(message):
    """Log terminal interaction message
    
    Args:
        message: Message to log
    """
    global _terminal_log, _full_terminal_output
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    _terminal_log.append(log_entry)
    _full_terminal_output.append(message)  # Store raw terminal output without timestamp
    print(message)  # Still print to terminal

def log_full_terminal_output(text):
    """Log full terminal output including agent interactions
    
    Args:
        text: Raw terminal output text
    """
    global _full_terminal_output
    if text:
        _full_terminal_output.append(text)

def _update_realtime_summary(item, attempt_result, execution_metadata):
    """Update realtime summary with flattened structure
    
    Args:
        item: Original dataset item
        attempt_result: Dict containing attempt details
        execution_metadata: Dict containing execution metadata
    """
    global _realtime_summary, _experiment_session
    
    if not _experiment_session:
        return
    
    item_id = str(item.get("id", item.get("unique_id", "unknown")))
    attempt_num = attempt_result.get("attempt", 1)
    
    # Initialize item in summary if not exists
    if item_id not in _realtime_summary:
        # Create flattened structure without nested batch_info and execution_metadata
        flattened_item = {
            **item,  # All original fields
            "model": execution_metadata.get("model", "unknown") if execution_metadata else "unknown",
            "provider": execution_metadata.get("provider", "unknown") if execution_metadata else "unknown", 
            "agent_framework": execution_metadata.get("agent_framework", "unknown") if execution_metadata else "unknown",
            "attempts_details": [],
            "total_attempts": 0,
            "successful_attempt": None,
            "total_token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        _realtime_summary[item_id] = flattened_item
    
    # Update attempt details
    attempt_detail = {
        "attempt": attempt_num,
        "result": attempt_result.get("result", False),
        "reason": attempt_result.get("reason", "unknown"),
        "token_usage": attempt_result.get("token_usage", {}),
        "history": attempt_result.get("history", "")
    }
    
    # Add or update attempt
    existing_attempts = _realtime_summary[item_id]["attempts_details"]
    existing_attempt_index = next((i for i, a in enumerate(existing_attempts) if a["attempt"] == attempt_num), None)
    
    if existing_attempt_index is not None:
        # Update existing attempt
        existing_attempts[existing_attempt_index] = attempt_detail
    else:
        # Add new attempt
        existing_attempts.append(attempt_detail)
    
    # Update summary fields
    _realtime_summary[item_id]["total_attempts"] = len(existing_attempts)
    
    # Update successful attempt
    successful_attempts = [a for a in existing_attempts if a["result"]]
    if successful_attempts:
        _realtime_summary[item_id]["successful_attempt"] = min(a["attempt"] for a in successful_attempts)
        _realtime_summary[item_id]["JSR"] = True
    else:
        _realtime_summary[item_id]["successful_attempt"] = None
        _realtime_summary[item_id]["JSR"] = False
    
    # Update total token usage
    total_prompt_tokens = sum(a.get("token_usage", {}).get("prompt_tokens", 0) for a in existing_attempts)
    total_completion_tokens = sum(a.get("token_usage", {}).get("completion_tokens", 0) for a in existing_attempts)
    total_tokens = sum(a.get("token_usage", {}).get("total_tokens", 0) for a in existing_attempts)
    
    _realtime_summary[item_id]["total_token_usage"] = {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens
    }
    
    # Save realtime summary to file
    _save_realtime_summary()

def _save_realtime_summary():
    """Save current realtime summary to JSON file"""
    global _realtime_summary, _experiment_session
    
    if not _experiment_session or not _realtime_summary:
        return
    
    try:
        experiment_dir = _experiment_session["experiment_dir"]
        summary_filename = "realtime_summary.json"
        summary_path = os.path.join(experiment_dir, summary_filename)
        print("=============>summary_path", os.path.abspath(summary_path))
        
        # Convert to list format for consistency with final output
        summary_list = list(_realtime_summary.values())
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_list, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"Save realtime summary error: {e}")

def finalize_experiment_session(results_list):
    """Finalize experiment session with actual results and rename files
    
    Args:
        results_list: List of (item, verification_success) tuples with final results
        
    Returns:
        str: Path to final JSON file
    """
    global _experiment_session, _terminal_log
    
    try:
        if not _experiment_session or not results_list:
            return None
            
        # Calculate success rate for logging (not used in folder name)
        successful_count = sum(1 for _, success in results_list if success)
        total_count = len(results_list)
        success_rate = int((successful_count / total_count) * 100) if total_count > 0 else 0
        
        # Generate final names
        sanitized_model_name = _experiment_session["sanitized_model_name"]
        start_timestamp = _experiment_session["start_timestamp"]
        
        # Extract input filename without extension
        input_basename = "unknown"
        if _experiment_session["input_filename"]:
            input_basename = os.path.splitext(os.path.basename(_experiment_session["input_filename"]))[0]
        
        final_experiment_name = f"{input_basename}_{sanitized_model_name}_{start_timestamp}"
        final_json_filename = f"{final_experiment_name}.json"  # JSON file with same name as folder
        
        # Create final experiment directory
        result_dir = _experiment_session["result_dir"]
        final_experiment_dir = os.path.join(result_dir, final_experiment_name)
        
        # Rename the temporary experiment directory to final name
        old_experiment_dir = _experiment_session["experiment_dir"]
        if os.path.exists(old_experiment_dir) and old_experiment_dir != final_experiment_dir:
            if os.path.exists(final_experiment_dir):
                # If final dir exists, move contents
                import shutil
                for item in os.listdir(old_experiment_dir):
                    shutil.move(os.path.join(old_experiment_dir, item), 
                              os.path.join(final_experiment_dir, item))
                os.rmdir(old_experiment_dir)
            else:
                os.rename(old_experiment_dir, final_experiment_dir)
        else:
            os.makedirs(final_experiment_dir, exist_ok=True)
        
        # Use realtime summary if available, otherwise build from results_list
        if _realtime_summary:
            # Use realtime summary data (already flattened)
            final_results = list(_realtime_summary.values())
        else:
            # Fallback: build flattened structure from results_list
            final_results = []
            execution_metadata = _experiment_session["execution_metadata"] or {}
            
            for item, verification_success in results_list:
                flattened_result = {
                    **item,  # All original fields
                    "model": execution_metadata.get("model", "unknown"),
                    "provider": execution_metadata.get("provider", "unknown"), 
                    "agent_framework": execution_metadata.get("agent_framework", "unknown"),
                    "JSR": verification_success,
                    # Add empty attempt details if not available
                    "attempts_details": [],
                    "total_attempts": 0,
                    "successful_attempt": None if not verification_success else 1,
                    "total_token_usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }
                final_results.append(flattened_result)
        
        # Save summary JSON file inside the experiment folder (flattened structure)
        summary_json_path = os.path.join(final_experiment_dir, final_json_filename)
        with open(summary_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        # Save full terminal output (replacing terminal_log.txt)
        log_terminal_interaction(f"实验结束，成功率: {successful_count}/{total_count} ({success_rate}%)")
        
        # Save complete terminal output with ANSI cleaning
        full_terminal_log_path = os.path.join(final_experiment_dir, f"{final_experiment_name}.txt")
        full_terminal_content = "\n".join(_full_terminal_output)
        clean_terminal_content = clean_ansi_sequences(full_terminal_content)
        
        with open(full_terminal_log_path, 'w', encoding='utf-8') as f:
            f.write(clean_terminal_content)
        
        print(f"汇总结果已保存: {summary_json_path}")
        print(f"实验文件夹: {final_experiment_dir}")
        print(f"完整终端日志: {full_terminal_log_path}")
        print(f"成功率: {successful_count}/{total_count} ({success_rate}%)")
        
        return summary_json_path
        
    except Exception as e:
        print(f"Finalize experiment session error: {e}")
        return None

def save_streaming_result(item, attempt_result, execution_metadata=None, input_filename=None):
    """Save streaming result after each attempt to experiment folder and update realtime summary
    
    Args:
        item: Original dataset item with all fields
        attempt_result: Dict containing attempt details (attempt, result, reason, etc.)
        execution_metadata: Dict containing agent_framework, provider, and model info
        input_filename: Path to input dataset file (used for filename prefix)
        
    Returns:
        str: Path to saved file
    """
    global _experiment_session
    
    try:
        if not _experiment_session:
            print("Warning: Experiment session not initialized, using fallback streaming save")
            return _save_streaming_result_fallback(item, attempt_result, execution_metadata, input_filename)
        
        # Use records directory for detailed attempt logs
        records_dir = _experiment_session["records_dir"]
        experiment_dir = _experiment_session["experiment_dir"]
        
        # Generate filename for streaming result
        item_id = item.get("id", item.get("unique_id", "unknown"))
        attempt_num = attempt_result.get("attempt", "unknown")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        
        filename = f"id{item_id}_attempt{attempt_num}_{timestamp}.json"
        
        # Extract agent interaction log and convert to history for realtime summary
        agent_interaction_log = attempt_result.get("agent_interaction_log", "")
        
        # Store agent interaction log directly as plain text in history field
        history = agent_interaction_log if agent_interaction_log else ""
        
        # Create attempt_result with history for realtime summary
        attempt_result_with_history = attempt_result.copy()
        attempt_result_with_history["history"] = history
        
        # Prepare streaming data without log fields (will be saved separately)
        attempt_result_clean = {k: v for k, v in attempt_result.items() if k not in ["agent_interaction_log", "post_execution_log"]}
        
        # Update realtime summary JSON with history included
        _update_realtime_summary(item, attempt_result_with_history, execution_metadata)
        
        # Continue with original streaming save logic...
        
        streaming_data = {
            "item_id": item_id,
            "attempt_info": attempt_result_clean,
            "timestamp": timestamp,
            "execution_metadata": execution_metadata or {},
            "original_item": item
        }
        
        # Save streaming result to records folder
        result_file_path = os.path.join(records_dir, filename)
        with open(result_file_path, 'w', encoding='utf-8') as f:
            json.dump(streaming_data, f, indent=2, ensure_ascii=False)
        
        # Save agent interaction log to separate txt file with same name
        agent_interaction_log = attempt_result.get("agent_interaction_log", "")
        post_execution_log = attempt_result.get("post_execution_log", "")
        
        if agent_interaction_log or post_execution_log:
            txt_filename = filename.replace('.json', '.txt')
            txt_file_path = os.path.join(records_dir, txt_filename)
            
            # Clean ANSI escape sequences from the logs
            clean_agent_log = clean_ansi_sequences(agent_interaction_log)
            clean_post_log = clean_ansi_sequences(post_execution_log)
            
            # Combine logs with proper formatting
            full_log = ""
            if clean_agent_log:
                full_log += clean_agent_log
            if clean_post_log:
                if full_log and not full_log.endswith('\n'):
                    full_log += '\n'
                full_log += clean_post_log
            
            with open(txt_file_path, 'w', encoding='utf-8') as f:
                f.write(full_log)
        
        attempt_status = "SUCCESS" if attempt_result.get("result") else "FAILED"
        log_terminal_interaction(f"保存尝试结果 [{attempt_status}]: ID{item_id} 尝试{attempt_num}")
        
        return result_file_path
        
    except Exception as e:
        print(f"Save streaming result error: {e}")
        return None

def _save_streaming_result_fallback(item, attempt_result, execution_metadata=None, input_filename=None):
    """Fallback streaming save function (original implementation)"""
    try:
        # Determine framework-specific directory
        framework_name = execution_metadata.get("agent_framework", "Unknown") if execution_metadata else "Unknown"
        
        # Map framework names to directory names
        framework_dir_map = {
            "ReactOSAgent": "ReactOSAgent",
            "MITRE_ATTCK_OSAgent": "MITRE_ATTCK_OSAgent", 
            "AutoGPT_OSAgent": "AutoGPT"
        }
        
        framework_dir = framework_dir_map.get(framework_name, framework_name)
        
        # Create streaming results directory
        streaming_dir = os.path.join("results", framework_dir, "streaming")
        os.makedirs(streaming_dir, exist_ok=True)
        
        # Generate filename for streaming result
        item_id = item.get("id", item.get("unique_id", "unknown"))
        attempt_num = attempt_result.get("attempt", "unknown")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        
        if input_filename:
            input_basename = os.path.splitext(os.path.basename(input_filename))[0]
            filename = f"{input_basename}_id{item_id}_attempt{attempt_num}_{timestamp}.json"
        else:
            filename = f"id{item_id}_attempt{attempt_num}_{timestamp}.json"
        
        # Prepare streaming data
        streaming_data = {
            "item_id": item_id,
            "attempt_info": attempt_result,
            "timestamp": timestamp,
            "execution_metadata": execution_metadata or {},
            "original_item": item
        }
        
        # Save streaming result
        result_file_path = os.path.join(streaming_dir, filename)
        with open(result_file_path, 'w', encoding='utf-8') as f:
            json.dump(streaming_data, f, indent=2, ensure_ascii=False)
        
        attempt_status = "SUCCESS" if attempt_result.get("result") else "FAILED"
        print(f"Streaming result saved [{attempt_status}]: {result_file_path}")
        return result_file_path
        
    except Exception as e:
        print(f"Save streaming result error: {e}")
        return None


