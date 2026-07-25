from tokenetics import Tokenetics


def test_finalize_extracts_text_and_runs_default_post_hoc_trim(stub_client, make_fake_response):
    tk = Tokenetics(client=stub_client)
    response = make_fake_response("Here's your answer: 42.\n\nLet me know if you have any questions!")
    stored = tk.finalize(response)
    assert stored == "Here's your answer: 42."


def test_finalize_with_no_response_stages_returns_extracted_text_unchanged(
    stub_client, make_fake_response
):
    tk = Tokenetics(response_stages=[], client=stub_client)
    response = make_fake_response("Here's your answer: 42.\n\nLet me know if you have any questions!")
    stored = tk.finalize(response)
    assert stored == "Here's your answer: 42.\n\nLet me know if you have any questions!"


def test_finalize_records_a_cost_logger_entry(stub_client, make_fake_response):
    tk = Tokenetics(client=stub_client)
    response = make_fake_response("Plain answer with nothing to trim.")
    tk.finalize(response)

    entries = tk.logger.entries  # type: ignore[attr-defined]
    assert len(entries) == 1
    assert entries[0].stage_name == "post_hoc_trim"
