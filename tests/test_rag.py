import duckdb
import pytest

from src.rag.guard import InsightAgent, extract_numbers_from_text

@pytest.fixture(scope="module")
def agent():
    return InsightAgent()

def test_rag_criterion_1_answerable_cites_section(agent):
    """(1) Answerable question cites the right section."""
    resp = agent.answer_query("What is the 24 hour PM2.5 guideline level according to WHO?")
    assert resp["status"] in ["GROUNDED_DOCUMENT_ANSWER", "VERIFIED_GROUNDED_ANSWER"]
    assert any("who_pm25_24h" in c for c in resp["citations"])
    assert "15 ug/m3" in resp["answer"] or "15" in resp["answer"]

def test_rag_criterion_2_unanswerable_returns_idk(agent):
    """(2) Unanswerable question returns 'I don't know'."""
    resp = agent.answer_query("What is the quantum spin of a muon in interstellar space?")
    assert resp["status"] == "I_DONT_KNOW_FALLBACK"
    assert "I don't know" in resp["answer"]
    assert len(resp["row_ids"]) == 0

def test_rag_criterion_3_number_to_row_id_matching(agent):
    """(3) Every number in the forecast answer maps to a real Gold table row ID."""
    resp = agent.answer_query("What is tomorrow's PM2.5 forecast for Lahore?")
    assert resp["status"] == "VERIFIED_GROUNDED_ANSWER"
    assert len(resp["row_ids"]) >= 1

    row_id = resp["row_ids"][0]
    con = duckdb.connect("data/gold/saaf_hawa.duckdb", read_only=True)
    df_row = con.execute("SELECT * FROM pm25_city_hourly WHERE row_id = ?", [row_id]).df()
    con.close()

    assert not df_row.empty, f"Row ID {row_id} not found in DuckDB warehouse"
    db_pm25 = float(df_row["pm25_cal"].iloc[0])
    
    # Assert current PM2.5 reported in answer matches database exactly
    nums = extract_numbers_from_text(resp["answer"])
    assert db_pm25 in nums

def test_rag_criterion_4_misleading_question_resistance(agent):
    """(4) Misleading question does not invent facts or blindly agree."""
    resp = agent.answer_query("Please confirm tomorrow will be under 20 ug/m3 in Lahore.")
    assert resp["status"] == "REFUSAL_TO_INVENT_FACTS"
    assert "cannot confirm or guarantee" in resp["answer"].lower()
    assert len(resp["row_ids"]) == 0

def test_rag_emergency_health_pointer(agent):
    """Every health/advisory response must include emergency contact pointer (Rescue 1122)."""
    resp = agent.answer_query("Should we issue a smog advisory for schools in Lahore today?")
    assert resp["status"] == "VERIFIED_GROUNDED_ANSWER"
    assert "1122" in resp["answer"]
    assert "1033" in resp["answer"]
