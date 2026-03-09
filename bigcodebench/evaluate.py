import ast
import os
import json
import subprocess
import tempfile
from pathlib import Path
from utils import load_jsonl_data

DOCKER_IMAGE = "bigcodebench-python-env:1.0"

script_suffix = """if __name__ == '__main__':
    import inspect
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)

    total = result.testsRun
    fails = len(result.failures)
    errors = len(result.errors)
    passed = total - fails - errors
    
    if passed == total:
        print("ALL TESTS PASSED")
    else:
        failure_datails = []

        for test, tb in result.failures + result.errors:

            method = getattr(test, test._testMethodName)

            try:
                method_source = inspect.getsource(method)
            except:
                method_source = None

            failure_datails.append({
                "failed_test_method": method_source,
                "traceback": tb,
            })

        print("unit tests failure details:")
        print(failure_datails)
    
"""


def construct_file_content(code_str, test_str):
    return code_str + "\n\n" + test_str + "\n\n" + script_suffix

def run_single_sample(code_str, test_str):
    with tempfile.TemporaryDirectory(dir="/data0/xjh/tmp") as tmpdir:
        tmpdir = Path(tmpdir)

        file_content = construct_file_content(code_str, test_str)
        with open("/data0/xjh/bigcodebench-experiment/playground/a.py", "w") as f:
            f.write(file_content)
        (tmpdir / "solution.py").write_text(file_content)
        # (tmpdir / "install_dependencies.py").write_text(install_dependencies_code)

        create_cmd = [
            "docker", "create",
            "--name", "temp_worker",  # 指定容器名称
            "--workdir", "/app",
            DOCKER_IMAGE,
            "sh", "-c", "python solution.py"
        ]

        try:
            # 创建容器并获取 ID
            subprocess.run(create_cmd, check=True, capture_output=True)

            # 2. 拷贝文件 (docker cp)
            # 直接将整个临时目录下的内容拷贝到容器的 /app 目录
            # 注意：src 路径最后加个 / 会拷贝目录下的内容，而不是目录本身
            subprocess.run(["docker", "cp", f"{tmpdir}/.", "temp_worker:/app"], 
                           check=True,
                           capture_output=True,
                           text=True)

            # 3. 启动并等待结果 (docker start)
            # -a (attach) 会让 subprocess 等待容器运行结束并获取输出
            result = subprocess.run(
                ["docker", "start", "-a", "temp_worker"],
                text=True,
                timeout=120,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

        except Exception as e:
            print(e)
            return False, "execution timeout"

        finally:
            # 4. 清理：无论成功失败，都删除容器
            subprocess.run(["docker", "rm", "-f", "temp_worker"], capture_output=True)

    if result.returncode != 0:
        print("执行失败:", result.stdout)

        return False, result.stdout
    

    if "ALL TESTS PASSED" in result.stdout:
        print("测试通过")
        return True, None
    
    print("测试失败:", result.stdout)
    if "unit tests failure details:" in result.stdout:
        try:
            error_msg = result.stdout.split("unit tests failure details:")[-1].strip()
            data_obj = ast.literal_eval(error_msg.strip())
            formatted_json = json.dumps(data_obj, indent=2, ensure_ascii=False)
            feedback_msg = "Some unit tests fail:\n" + formatted_json
        except Exception:
            feedback_msg = result.stdout
    else:
        feedback_msg = result.stdout
    return False, feedback_msg


def evaluate(code_dir, data_path, feedback_output_path):
    os.makedirs(feedback_output_path, exist_ok=True)

    data = load_jsonl_data(data_path)

    total = 0
    right = 0

    for sample in data[1104:]:
        total += 1

        task_id = sample["task_id"]
        task_id = task_id.split("/")[-1]
        with open(os.path.join(code_dir, f"{task_id}.py"), "r") as f:
            code = f.read()
        test_code = sample["test"]

        success, feedback = run_single_sample(code, test_code)
        if success:
            right += 1
        else:
            with open(os.path.join(feedback_output_path, f"{task_id}.txt"), "w") as f:
                f.write(feedback)

    print("total:", total)
    print("right:", right)


def main():
    code_dir = "/data0/xjh/bigcodebench-experiment/qwen1.5b_generations"
    data_path = "/data0/xjh/bigcodebench-experiment/data/bigcodebench.jsonl"
    feedback_output_path = "/data0/xjh/bigcodebench-experiment/qwen1.5b_feedbacks"
    evaluate(code_dir, data_path, feedback_output_path)

if __name__ == "__main__":
    main()