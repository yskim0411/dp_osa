import subprocess

# Value Setting
nb_teachers = 50
nb_labels = 2
stdnt_share_list = [50]
lap_scale_list = [10]
moments = 10

# dataset 설정
dataset = "remosa_tab"
model = "rf"

# train_student.py 실행 후 analysis.py 실행
for stdnt_share in stdnt_share_list:  # stdnt_share 반복
    for lap_scale in lap_scale_list:  # lap_scale 반복
        # train_student.py 명령어 실행
        train_command = [
            "python", 
            "train_student.py", 
            f"--nb_teachers={nb_teachers}", 
            f"--nb_labels={nb_labels}",
            f"--dataset={dataset}",
            f"--model={model}",
            f"--lap_scale={lap_scale}",
            f"--stdnt_share={stdnt_share}",
            # "--deeper", "True",
            # "--max_steps", "3000"
        ]

        print(f"query: {stdnt_share}, lap: {lap_scale}")
        
        result = subprocess.run(train_command)
        if result.returncode != 0:
            print(f"Command {train_command} failed with return code {result.returncode}")
            continue
        else:
            print(f"Command {train_command} executed successfully")

        # noise_eps를 lap_scale에 따라 동적으로 설정
        noise_eps = 2 / lap_scale

        # analysis.py 명령어 실행
        counts_file = f"/home/aix21202/pate/privacy/research/pate_2017/tmp/{dataset}_{nb_teachers}_student_clean_votes_lap_{lap_scale}.npy"
        analysis_command = [
            "python", 
            "analysis.py", 
            f"--counts_file={counts_file}",
            f"--noise_eps={noise_eps}",
            "--input_is_counts=True", 
            f"--max_examples={stdnt_share}",
            f"--moments={moments}",
        ]
        
        analysis_result = subprocess.run(analysis_command)
        if analysis_result.returncode != 0:
            print(f"Command {analysis_command} failed with return code {analysis_result.returncode}")
        else:
            print(f"Command {analysis_command} executed successfully")
            
        print("=====================================================")
