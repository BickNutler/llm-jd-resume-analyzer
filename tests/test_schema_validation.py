from src.api.schemas import AnalyzeResponse, BaselineOut

def test_response_schema_validates():
    resp = AnalyzeResponse(
        match_score=0.5,
        baseline=BaselineOut(tfidf_score=0.1, keyword_coverage=0.9, missing_keywords=[], matched_keywords=[]),
        skills_resume={"skills":[],"tools":[],"requirements":[]},
        skills_job={"skills":[],"tools":[],"requirements":[]},
        missing_keywords=[],
        recommended_keywords=[],
        explanations={"llm_enabled": False},
    )
    assert resp.match_score == 0.5
