from kvbridge.evaluation import extract_numeric_answer


def test_extracts_gsm8k_tagged_answer():
    assert extract_numeric_answer("Work shown here. #### 1,024") == "1024"
