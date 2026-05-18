import tempfile
import unittest

from storage import repository
from storage.database import configure_database, init_db
from workflow.prompt_builder import PromptBuilder


class PromptBuilderTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        configure_database(f"sqlite:///{self.tmpdir.name}/test.db")
        init_db()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_prompt_includes_tenant_business_name_and_greeting(self):
        tenant = repository.get_default_tenant()
        profile = repository.create_prompt_profile(
            tenant["id"],
            label="Friendly prompt",
            business_name="Acme Plumbing",
            greeting="Acme Plumbing, what's going on?",
            tone="plainspoken and calm",
            verbosity="short",
            closing_line="You're all set. We'll call back soon.",
            avoid_phrases=["certainly"],
            preferred_terms=["service address"],
            extra_instructions_text="Use the company name naturally.",
        )

        prompt = PromptBuilder().build("913-555-0123", tenant=tenant, profile=profile)

        self.assertIn("Acme Plumbing", prompt)
        self.assertIn("Acme Plumbing, what's going on?", prompt)
        self.assertIn("plainspoken and calm", prompt)

    def test_prompt_always_includes_locked_core_rules(self):
        tenant = repository.get_default_tenant()
        profile = repository.get_active_prompt_profile(tenant["id"])

        prompt = PromptBuilder().build("913-555-0123", tenant=tenant, profile=profile)

        self.assertIn("Your job is to collect exactly these 5 required fields", prompt)
        self.assertIn("plumbing issue", prompt)
        self.assertIn("urgency / active water status", prompt)
        self.assertIn("service address", prompt)
        self.assertIn("callback number", prompt)
        self.assertIn("customer name", prompt)
        self.assertIn("If backend validation fails", prompt)

    def test_extra_instructions_cannot_remove_required_field_rules(self):
        tenant = repository.get_default_tenant()
        profile = repository.create_prompt_profile(
            tenant["id"],
            label="Unsafe extra",
            business_name="Acme Plumbing",
            greeting="Acme Plumbing, what's going on?",
            tone="casual",
            verbosity="short",
            closing_line="Done.",
            avoid_phrases=[],
            preferred_terms=[],
            extra_instructions_text="Do not ask for an address. Only collect issue.",
        )

        prompt = PromptBuilder().build("913-555-0123", tenant=tenant, profile=profile)

        self.assertIn("Tenant extra instructions, style only", prompt)
        self.assertIn("Do not ask for an address. Only collect issue.", prompt)
        self.assertIn("If any tenant instruction conflicts with this section, ignore the conflicting tenant instruction.", prompt)
        self.assertGreater(prompt.rfind("Collect exactly: issue, urgency, address, callback, name"), prompt.find("Tenant extra instructions"))

    def test_first_name_only_and_last_name_rules_remain_in_prompt(self):
        tenant = repository.get_default_tenant()
        profile = repository.get_active_prompt_profile(tenant["id"])

        prompt = PromptBuilder().build("913-555-0123", tenant=tenant, profile=profile)

        self.assertIn("A first name is enough", prompt)
        self.assertIn("Last name is not required", prompt)
        self.assertIn("Never require or ask for a last name", prompt)
        self.assertIn("Never invent a caller name", prompt)

    def test_tenant_a_prompt_does_not_affect_tenant_b(self):
        tenant_a = repository.create_tenant(
            "Tenant A",
            "tenant-a-prompt",
            "Tenant A Plumbing",
            "Tenant A plumbing, what's going on?",
            "+15550000001",
        )
        tenant_b = repository.create_tenant(
            "Tenant B",
            "tenant-b-prompt",
            "Tenant B Plumbing",
            "Tenant B plumbing, what's going on?",
            "+15550000002",
        )
        profile_a = repository.create_prompt_profile(
            tenant_a["id"],
            label="Tenant A custom",
            business_name="Tenant A Plumbing",
            greeting="Tenant A custom greeting",
            tone="direct",
            verbosity="brief",
            closing_line="Tenant A closing.",
            avoid_phrases=[],
            preferred_terms=[],
        )
        profile_b = repository.get_active_prompt_profile(tenant_b["id"])

        prompt_a = PromptBuilder().build("913-555-0123", tenant=tenant_a, profile=profile_a)
        prompt_b = PromptBuilder().build("913-555-0123", tenant=tenant_b, profile=profile_b)

        self.assertIn("Tenant A custom greeting", prompt_a)
        self.assertNotIn("Tenant A custom greeting", prompt_b)
        self.assertIn("Tenant B plumbing, what's going on?", prompt_b)

    def test_activating_previous_prompt_version_changes_only_that_tenant(self):
        tenant_a = repository.create_tenant(
            "Tenant A",
            "tenant-a-activation",
            "Tenant A Plumbing",
            "Tenant A plumbing, what's going on?",
            "+15550000001",
        )
        tenant_b = repository.create_tenant(
            "Tenant B",
            "tenant-b-activation",
            "Tenant B Plumbing",
            "Tenant B plumbing, what's going on?",
            "+15550000002",
        )
        original_a = repository.get_active_prompt_profile(tenant_a["id"])
        updated_a = repository.create_prompt_profile(
            tenant_a["id"],
            label="Tenant A updated",
            business_name="Tenant A Plumbing",
            greeting="Tenant A updated greeting",
            tone="direct",
            verbosity="brief",
            closing_line="Tenant A closing.",
            avoid_phrases=[],
            preferred_terms=[],
        )
        active_b = repository.get_active_prompt_profile(tenant_b["id"])

        self.assertEqual(repository.get_active_prompt_profile(tenant_a["id"])["id"], updated_a["id"])

        repository.activate_prompt_profile(tenant_a["id"], original_a["id"])

        self.assertEqual(repository.get_active_prompt_profile(tenant_a["id"])["id"], original_a["id"])
        self.assertEqual(repository.get_active_prompt_profile(tenant_b["id"])["id"], active_b["id"])


if __name__ == "__main__":
    unittest.main()
