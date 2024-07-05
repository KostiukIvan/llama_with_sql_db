import pandas as pd
import csv
from sql_manager import Manager
from tqdm import tqdm

prompt = lambda x : f"""1. Please treat date columns as strings without using strftime for any formatting.
Question: {x}"""

table_file_path = "data/fake_estatements.csv"
question_file_path = 'data/questions.csv'
output_file_path = "sql_poc/test_results.csv"
def test_manager_performance():
    manager = Manager(file_path=table_file_path)
    # load test data/questions
    question_df = pd.read_csv(question_file_path, delimiter=',')
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(["Question", "SQL", "Result"])
        for question in tqdm(question_df["Question"]):
            sql = manager.query_llm(query=prompt(question))
            result = manager.query_db(sql=sql)
            writer.writerow([question, sql, result])


def get_test_summary():
    df = pd.read_csv(output_file_path, delimiter='\t', encoding='latin1')
    fails_count = 0
    for res in df["Result"]:
        if type(res) is str and "None" in res:
            fails_count += 1
    print(f"Accuracy : {(len(df) - fails_count) / len(df) }")
    print(f"Fails={fails_count}, total={len(df)}")


def fine_tune_model():
    init_promt = lambda x, y, z: f"""Here is the question of the client "{x}". You generated the SQL query "{y}". It's not runnable! 
                                    Generate the new, correct SQL query. Here is the database previous result "{z}"."""
    input_file_path = "sql_poc/test_results.csv"
    runnable_sql_file_path = "sql_poc/runnable_sql_results.csv"
    failed_sql_file_path = "sql_poc/failed_sql_results.csv"
    manager = Manager(file_path=table_file_path, temperature=0.3)
    df = pd.read_csv(input_file_path, delimiter='\t', encoding='latin1')
    
    with open(runnable_sql_file_path, mode='w', newline='') as r_file:   
        with open(failed_sql_file_path, mode='w', newline='') as f_file:
            r_writer = csv.writer(r_file, delimiter='\t')
            f_writer = csv.writer(f_file, delimiter='\t')
            r_writer.writerow(["Question", "SQL", "Result"])
            f_writer.writerow(["Question", "SQL", "Result"])
            for i in range(0, len(df)):
                data = df.iloc[i]
                question = data["Question"]
                sql = data["SQL"]
                result = data["Result"]
                for _ in range(0, 5):
                    sql = manager.query_llm(query=prompt(question))
                    result = manager.query_db(sql=sql)
                    print(sql, result)
                    if type(result) is str and ("(0,)" in result or "None" in result):
                        # repeat
                        pass
                    else:
                        r_writer.writerow([question, sql, result])
                        break
                else:
                    print(f"Was not able to find the right SQL for Question={question}, SQL={sql}, Result={result}")
                    f_writer.writerow([question, sql, result])



if __name__=="__main__":
    test_manager_performance()
    # get_test_summary()
    # fine_tune_model()