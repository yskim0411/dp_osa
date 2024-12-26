import subprocess
import sys

# 설정 값들
nb_teachers = 1
nb_labels = 2
dataset = "posa"
model = "xgb"

precisions = []
aurocs = []

# 명령어 실행 및 결과 수집
for teacher_id in range(nb_teachers):
    command = [
        "python", 
        "train_teachers.py",
        f"--nb_teachers={nb_teachers}", 
        f"--teacher_id={teacher_id}", 
        f"--dataset={dataset}",
        f"--model={model}",
        f"--nb_labels={nb_labels}"
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout_lines = []

    # 실시간으로 출력 읽기
    for line in process.stdout:
        sys.stdout.write(line)
        stdout_lines.append(line)
    
    process.wait()

    if process.returncode != 0:
        print(f"Command {command} failed with return code {process.returncode}")
        continue

    # 정확도 및 AUROC 추출
    try:
        accuracy = next(float(line.split(":")[-1].strip().replace('%', '')) for line in stdout_lines if "Model accuracy:" in line)
        precisions.append(accuracy)
        print(f"Teacher {teacher_id} - Accuracy: {accuracy}%")

        auroc = next(float(line.split(":")[-1].strip()) for line in stdout_lines if "Model AUROC:" in line)
        aurocs.append(auroc)
        print(f"Teacher {teacher_id} - AUROC: {auroc}")
        
        print("====================================")
        print()

    except (StopIteration, ValueError):
        print(f"Failed to extract precision or AUROC for Teacher {teacher_id}.")

# 평균 계산 및 출력
if precisions:
    average_precision = sum(precisions) / len(precisions)
    print(f"Average precision: {average_precision:.2f}%")

if aurocs:
    average_auroc = sum(aurocs) / len(aurocs)
    print(f"Average AUROC: {average_auroc:.2f}")