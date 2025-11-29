import sys

def log_print(*args, **kwargs):
    """
    自定义打印函数，同时输出到 stdout 和日志文件。
    """
    original_print(*args, **kwargs)
    if _log_file is not None:
        message = ' '.join(str(arg) for arg in args)
        _log_file.write(message + '\n')
        _log_file.flush()

def set_log_file(log_file_path):
    """设置日志文件路径。"""
    global _log_file
    if log_file_path:
        try:
            _log_file = open(log_file_path, 'w', encoding='utf-8')
            return True
        except Exception as e:
            print(f"Error opening log file {log_file_path}: {e}")
            return False
    return True

def close_log_file():
    """关闭日志文件。"""
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None

def read_file_content(filepath):
    """
    读取并返回文件内容。
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        sys.exit(1)

def extract_detailed_solution(solution, marker='Detailed Solution', after=True):
    """
    从解决方案字符串中提取 '### Detailed Solution ###' 之后或之前的文本。
    """
    idx = solution.find(marker)
    if idx == -1:
        return ''
    if after:
        return solution[idx + len(marker):].strip()
    else:
        return solution[:idx].strip()

def verify_solution(problem_statement, solution, verbose=True):
    dsol = extract_detailed_solution(solution)
    newst = f"""
======================================================================
### Problem ###

{problem_statement}

======================================================================
### Solution ###

{dsol}

{verification_remider}
"""
    if verbose:
        print(">>>>>>> Start verification.")
    
    contents1 = [{"role": "user", "parts": [{"text": newst}]}]
    out = call_gemini_api(
        system_instruction=verification_system_prompt, 
        contents=contents1,
        verbose=verbose
    )
    if not out:
        print(">>>>>>> Verification call failed.")
        return "", "no"

    if verbose:
        print(">>>>>>> Verification results:")
        print(out)

    check_correctness = f'Response in "yes" or "no". Is the following statement saying the solution is correct, or does not contain critical error or a major justification gap?\n\n{out}'
    contents2 = [{"role": "user", "parts": [{"text": check_correctness}]}]
    o = call_gemini_api(
        system_instruction=None,
        contents=contents2,
        verbose=verbose
    )
    if not o:
        print(">>>>>>> Verification check call failed.")
        return "", "no"

    if verbose:
        print(">>>>>>> Is verification good?")
        print(o)
        
    bug_report = ""
    if "yes" not in o.lower():
        bug_report = extract_detailed_solution(out, "Detailed Verification", False)

    if verbose:
        print(">>>>>>> Bug report:")
        print(bug_report)
    
    return bug_report, o

def check_if_solution_claimed_complete(solution):
    check_complete_prompt = f"""
Is the following text claiming that the solution is complete?
==========================================================

{solution}

==========================================================

Response in exactly "yes" or "no". No other words.
    """
    contents = [{"role": "user", "parts": [{"text": check_complete_prompt}]}]
    o = call_gemini_api(
        system_instruction=None,
        contents=contents,
        verbose=False # This is a simple check, no need for verbose output
    )
    if not o:
        return False
        
    print(o)
    return "yes" in o.lower()    

def init_explorations(problem_statement, verbose=True, other_prompts=[]):
    contents = [{"role": "user", "parts": [{"text": problem_statement}]}]
    if other_prompts:
        for prompt in other_prompts:
            contents.append({"role": "user", "parts": [{"text": prompt}]})

    print(">>>>>> Initial prompt.")
    output1 = call_gemini_api(
        system_instruction=step1_prompt,
        contents=contents,
        verbose=verbose
    )
    if not output1:
        print(">>>>>> Initial generation failed.")
        return None, None, None, None
        
    print(">>>>>>> First solution: ") 
    print(output1)

    print(">>>>>>> Self improvement start:")
    contents.append({"role": "model", "parts": [{"text": output1}]})
    contents.append({"role": "user", "parts": [{"text": self_improvement_prompt}]})
    
    solution = call_gemini_api(
        system_instruction=step1_prompt,
        contents=contents,
        verbose=verbose
    )
    if not solution:
        print(">>>>>> Self-improvement generation failed.")
        return None, None, None, None

    print(">>>>>>> Corrected solution: ")
    print(solution)
    
    print(">>>>>>> Check if solution is complete:")
    is_complete = check_if_solution_claimed_complete(solution)
    if not is_complete:
        print(">>>>>>> Solution is not complete. Failed.")
        return None, None, None, None
    
    print(">>>>>>> Verify the solution.")
    verify, good_verify = verify_solution(problem_statement, solution, verbose)

    print(">>>>>>> Initial verification: ")
    print(verify)
    print(f">>>>>>> verify results: {good_verify}")
    
    return True, solution, verify, good_verify # Return a success flag instead of p1

def agent(problem_statement, other_prompts=[]):
    success_flag, solution, verify, good_verify = init_explorations(problem_statement, True, other_prompts)

    if not success_flag or solution is None:
        print(">>>>>>> Failed in finding a complete solution during initialization.")
        return None

    error_count = 0
    correct_count = 1 if "yes" in good_verify.lower() else 0
    
    for i in range(30):
        print(f"Number of iterations: {i}, number of corrects: {correct_count}, number of errors: {error_count}")

        if "yes" not in good_verify.lower():
            correct_count = 0
            error_count += 1

            print(">>>>>>> Verification does not pass, correcting ...")
            
            # 建立一个新的对话历史用于修正
            correction_contents = [{"role": "user", "parts": [{"text": problem_statement}]}]
            if other_prompts:
                 for prompt in other_prompts:
                    correction_contents.append({"role": "user", "parts": [{"text": prompt}]})
            
            correction_contents.append({"role": "model", "parts": [{"text": solution}]})
            correction_contents.append({"role": "user", "parts": [{"text": correction_prompt}, {"text": verify}]})

            print(">>>>>>> New correction prompt being sent.")
            new_solution = call_gemini_api(
                system_instruction=step1_prompt,
                contents=correction_contents,
                verbose=True
            )
            
            if not new_solution:
                print(">>>>>>> Correction attempt failed to generate a response. Stopping.")
                return None
            
            solution = new_solution
            print(">>>>>>> Corrected solution:")
            print(solution)

            print(">>>>>>> Check if new solution is complete:")
            is_complete = check_if_solution_claimed_complete(solution)
            if not is_complete:
                print(">>>>>>> Solution is not complete after correction. Failed.")
                return None
        
        print(">>>>>>> Verify the solution.")
        verify, good_verify = verify_solution(problem_statement, solution)

        if "yes" in good_verify.lower():
            print(">>>>>>> Solution is good, verifying again ...")
            correct_count += 1
            error_count = 0
        else:
             # 如果验证再次失败，重置正确计数器
             correct_count = 0

        if correct_count >= 5:
            print(">>>>>>> Correct solution found and verified multiple times.")
            print(solution)
            return solution
        elif error_count >= 10:
            print(">>>>>>> Failed to find a correct solution after multiple errors.")
            return None

    print(">>>>>>> Failed to find a correct solution within the iteration limit.")
    return None