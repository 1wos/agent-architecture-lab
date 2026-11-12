"""모델 호출 없이 공개 함수의 제어 흐름과 계약을 확인한다. AI 품질 평가는 별도다."""
import unittest

from retro_demo import Analysis, DemoBackend, Retro, Review, run_retro


def entry(**overrides):
    data = {"anonymous_id": "anon-test", "data_type": "synthetic", "week": 6,
            "keep": "테스트를 작성했습니다.", "problem": "인용이 틀렸습니다.", "try": "검사를 추가하겠습니다."}
    return Retro.model_validate(data | overrides)


class RetroChecks(unittest.TestCase):
    def test_normal_feedback_uses_all_original_fields(self):
        retro = entry()
        result = run_retro(retro, DemoBackend())
        self.assertEqual(result["data"]["outcome"], "ready")
        self.assertEqual(result["data"]["revisions"], 0)
        self.assertEqual(result["data"]["metrics"]["api_calls"], 0)
        self.assertEqual(result["data"]["metrics"]["role_calls"], 3)
        for item in result["data"]["feedback"]["items"]:
            self.assertIn(item["quote"], retro.text(item["field"]))

    def test_fake_quote_is_repaired_once(self):
        result = run_retro(entry(), DemoBackend("repair"))
        self.assertIn("원문에 없는 인용", result["data"]["trace"][1]["issues"][0])
        self.assertEqual(result["data"]["revisions"], 1)
        self.assertEqual(result["data"]["outcome"], "ready")

    def test_persistent_bad_quote_holds_feedback(self):
        backend = DemoBackend("blocked")
        result = run_retro(entry(), backend)
        self.assertEqual(result["data"]["outcome"], "needs_review")
        self.assertIsNone(result["data"]["feedback"])
        self.assertEqual(result["data"]["revisions"], 1)
        self.assertEqual(backend.calls, 3)

    def test_empty_retro_stops_before_any_call(self):
        backend = DemoBackend()
        result = run_retro(entry(keep=" ", problem="", **{"try": ""}), backend)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "empty_retro")
        self.assertEqual(backend.calls, 0)

    def test_partial_retro_does_not_invent_missing_fields(self):
        result = run_retro(entry(keep="", problem=""), DemoBackend())
        self.assertEqual([i["field"] for i in result["data"]["feedback"]["items"]], ["try"])

    def test_long_valid_retro_still_returns_bounded_feedback(self):
        result = run_retro(entry(keep="가" * 2000), DemoBackend())
        self.assertEqual(result["data"]["outcome"], "ready")
        self.assertLessEqual(len(result["data"]["feedback"]["items"][0]["quote"]), 160)

    def test_reviewer_requests_revision_then_accepts(self):
        class CriticalReviewer(DemoBackend):
            reviews = 0

            def call(self, role, payload, schema):
                result = super().call(role, payload, schema)
                if role == "reviewer":
                    self.reviews += 1
                    return Review(issues=["행동을 더 작게 나눠 주세요"] if self.reviews == 1 else [])
                return result
        backend = CriticalReviewer()
        result = run_retro(entry(), backend)
        self.assertEqual(result["data"]["outcome"], "ready")
        self.assertEqual(result["data"]["revisions"], 1)
        self.assertEqual(backend.calls, 5)

    def test_reviewer_rejects_both_drafts_without_third_attempt(self):
        class AlwaysRejects(DemoBackend):
            def call(self, role, payload, schema):
                result = super().call(role, payload, schema)
                return Review(issues=["근거와 의미가 맞지 않습니다"]) if role == "reviewer" else result
        backend = AlwaysRejects()
        result = run_retro(entry(), backend)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["data"]["outcome"], "needs_review")
        self.assertIsNone(result["data"]["feedback"])
        self.assertEqual(backend.calls, 5)

    def test_analyst_invented_quote_stops_pipeline(self):
        class BadAnalyst(DemoBackend):
            def call(self, role, payload, schema):
                result = super().call(role, payload, schema)
                if isinstance(result, Analysis):
                    result.observations[0].quote = "원문에 없는 행동"
                return result
        backend = BadAnalyst()
        result = run_retro(entry(), backend)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(backend.calls, 1)

    def test_provider_failure_stops_without_leaking_exception_message(self):
        class BrokenBackend(DemoBackend):
            def call(self, role, payload, schema):
                self.calls += 1
                raise RuntimeError("private request details")
        backend = BrokenBackend()
        result = run_retro(entry(), backend)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "RuntimeError")
        self.assertIsNone(result["data"]["feedback"])
        self.assertEqual(backend.calls, 1)

    def test_schema_rejects_invalid_week_and_extra_fields(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            entry(week=9)
        with self.assertRaises(ValidationError):
            entry(unexpected="text")


if __name__ == "__main__":
    unittest.main()
