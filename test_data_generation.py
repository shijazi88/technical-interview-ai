#!/usr/bin/env python3
"""
Quick test for data generation to ensure no infinite loops
Run this before starting full training to verify everything works
"""

import time
from technical_questions_db import TechnicalQuestionsDatabase
from enhanced_data_processor import TechnicalInterviewDataset

def test_data_generation():
    """Test data generation with timing"""
    print("🧪 Testing Data Generation System")
    print("=" * 50)
    
    # Initialize database
    print("📚 Loading questions database...")
    start_time = time.time()
    questions_db = TechnicalQuestionsDatabase()
    print(f"✅ Loaded {len(questions_db.questions)} questions in {time.time() - start_time:.1f}s")
    
    # Test single interview generation
    print("\n🎯 Testing single interview generation...")
    dataset_creator = TechnicalInterviewDataset(questions_db)
    
    start_time = time.time()
    single_interview = dataset_creator._generate_single_interview()
    generation_time = time.time() - start_time
    
    print(f"✅ Generated single interview with {len(single_interview)} exchanges")
    print(f"⏱️ Time taken: {generation_time:.2f} seconds")
    
    if generation_time > 5.0:
        print("⚠️ WARNING: Single interview took longer than expected (>5s)")
        print("This might cause issues with larger datasets")
    else:
        print("🚀 Generation speed looks good!")
    
    # Test small batch
    print(f"\n📊 Testing batch generation (3 scenarios)...")
    start_time = time.time()
    batch_scenarios = dataset_creator.create_realistic_interview_scenarios(3)
    batch_time = time.time() - start_time
    
    print(f"✅ Generated {len(batch_scenarios)} total examples from 3 scenarios")
    print(f"⏱️ Total time: {batch_time:.2f} seconds")
    print(f"📈 Average per scenario: {batch_time/3:.2f} seconds")
    
    # Analyze generated data
    print(f"\n📋 Sample Analysis:")
    if batch_scenarios:
        sample = batch_scenarios[0]
        print(f"- Instruction length: {len(sample['instruction'])} chars")
        print(f"- Input length: {len(sample['input'])} chars") 
        print(f"- Output length: {len(sample['output'])} chars")
        print(f"- Language: {sample['metadata']['language']}")
        print(f"- Experience: {sample['metadata']['experience_level']}")
        print(f"- Category: {sample['metadata']['category']}")
    
    # Performance assessment
    expected_time_per_scenario = batch_time / 3
    estimated_time_20 = expected_time_per_scenario * 20
    estimated_time_150 = expected_time_per_scenario * 150
    
    print(f"\n⏰ Time Estimates:")
    print(f"- 20 scenarios: ~{estimated_time_20/60:.1f} minutes")
    print(f"- 150 scenarios: ~{estimated_time_150/60:.1f} minutes")
    
    if estimated_time_20 > 300:  # 5 minutes
        print("⚠️ WARNING: 20 scenarios might take longer than 5 minutes")
        print("Consider reducing the number for initial testing")
    
    print(f"\n🎉 Data generation test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_data_generation()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("Please fix the issue before running full training")
        import traceback
        traceback.print_exc() 