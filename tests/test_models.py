"""Test that our data models work correctly."""

from src.data_models import (
    UseCaseCategory,
    SectionType,
    Requirement,
    UsecaseSection,
    DocumentChunk,
    ServiceFlowStep,
    UseCaseSummary,
    get_use_case_category,
    get_parent_id,
)


def test_enums():
    """Test enum values."""
    print("Testing enums...")
    
    assert UseCaseCategory.CONSUMER.value == "consumer"
    assert UseCaseCategory.BUSINESS.value == "business"
    assert SectionType.DESCRIPTION.value == "description"
    
    print("  ✓ Enums work")


def test_requirement():
    """Test Requirement model."""
    print("Testing Requirement...")
    
    req = Requirement(
        requirement_id="PR 5.1.1-1",
        use_case_id="5.1.1",
        text="The mobile network is used to contact AI-Core...",
        page_number=11
    )
    
    assert req.requirement_id == "PR 5.1.1-1"
    assert req.use_case_id == "5.1.1"
    
    print("  ✓ Requirement model works")


def test_usecase_section():
    """Test UsecaseSection model."""
    print("Testing UsecaseSection...")
    
    section = UsecaseSection(
        section_id="5.1.1.1",
        parent_id="5.1.1",
        title="Description",
        level=4,
        section_type=SectionType.DESCRIPTION,
        content="The growth in the use of AI agents...",
        page_start=10,
        page_end=11,
        requirements=[]
    )
    
    assert section.section_id == "5.1.1.1"
    assert section.level == 4
    assert section.section_type == SectionType.DESCRIPTION
    
    print("  ✓ UsecaseSection model works")


def test_document_chunk():
    """Test DocumentChunk model."""
    print("Testing DocumentChunk...")
    
    chunk = DocumentChunk(
        chunk_id="chunk_5.1.1.1_001",
        content="The growth in the use of AI agents...",
        use_case_id="5.1.1",
        section_id="5.1.1.1",
        section_type=SectionType.DESCRIPTION,
        category=UseCaseCategory.CONSUMER,
        page_start=10,
        page_end=11,
        requirement_codes=["PR 5.1.1-1", "PR 5.1.1-2"],
        token_count=150
    )
    
    assert chunk.category == UseCaseCategory.CONSUMER
    assert len(chunk.requirement_codes) == 2
    
    print("  ✓ DocumentChunk model works")


def test_usecase_summary():
    """Test UseCaseSummary model."""
    print("Testing UseCaseSummary...")
    
    summary = UseCaseSummary(
        use_case_id="5.1.1",
        title="AI Agents to Enable Smart Life",
        category=UseCaseCategory.CONSUMER,
        actors=["User", "Robot-servant", "Smart car"],
        network_functions=["AMF", "SMF", "PCF"],
        description="AI agents coordinate smart devices for daily life.",
        service_flow=[
            ServiceFlowStep(
                step_number=1,
                description="Robot-servant sends intent to network",
                actors_involved=["Robot-servant", "AI Agent"]
            )
        ],
        requirement_ids=["PR 5.1.1-1", "PR 5.1.1-2"],
        source_pages=[10, 11]
    )
    
    assert summary.use_case_id == "5.1.1"
    assert len(summary.actors) == 3
    assert len(summary.service_flow) == 1
    
    print("  ✓ UseCaseSummary model works")


def test_helper_functions():
    """Test helper functions."""
    print("Testing helper functions...")
    
    # Test get_use_case_category
    assert get_use_case_category("5.1.1") == UseCaseCategory.CONSUMER
    assert get_use_case_category("5.2.3") == UseCaseCategory.BUSINESS
    assert get_use_case_category("5.3.1") == UseCaseCategory.OPERATOR
    
    # Test get_parent_id
    assert get_parent_id("5.1.1.1") == "5.1.1"
    assert get_parent_id("5.1.1") == "5.1"
    assert get_parent_id("5") is None
    
    print("  ✓ Helper functions work")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Testing Data Models")
    print("="*50 + "\n")
    
    test_enums()
    test_requirement()
    test_usecase_section()
    test_document_chunk()
    test_usecase_summary()
    test_helper_functions()
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50 + "\n")