import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import MODEL_ID, MODEL_DIR, DATA_DIR, TRAINING_DIR

DATA_FILE = os.path.join(DATA_DIR, "math_train.jsonl")


def generate_training_data():
    """Generate math training data with proofs and step-by-step solutions."""
    # Also generate code-generation training data
    try:
        from code_training_data import generate_code_training_data
        generate_code_training_data()
    except ImportError:
        pass

    samples = [
        {
            "problem": "Find the derivative of f(x) = 3x^3 + 2x^2 - 5x + 1",
            "solution": "f'(x) = 9x^2 + 4x - 5",
            "steps": [
                {"step": "d/dx(3x^3) = 9x^2", "explanation": "Power rule: d/dx(ax^n) = a*n*x^(n-1)", "proof": "Power Rule: d/dx(x^n) = n*x^(n-1) by first principles of differentiation"},
                {"step": "d/dx(2x^2) = 4x", "explanation": "Power rule with a=2, n=2", "proof": "Power Rule"},
                {"step": "d/dx(-5x) = -5", "explanation": "d/dx(ax) = a, since ax^1 -> a*1*x^0 = a", "proof": "Power Rule with n=1"},
                {"step": "d/dx(1) = 0", "explanation": "Derivative of a constant is 0", "proof": "Constant Rule: d/dx(c) = 0"}
            ],
            "proofs": [
                {"theorem": "Power Rule", "statement": "d/dx(x^n) = n*x^(n-1)", "applied_to": "each term"},
                {"theorem": "Sum Rule", "statement": "d/dx(f+g) = f' + g'", "applied_to": "all terms"},
                {"theorem": "Constant Rule", "statement": "d/dx(c) = 0", "applied_to": "constant term 1"}
            ]
        },
        {
            "problem": "Evaluate the integral ∫(2x + 3) dx from 0 to 2",
            "solution": "6",
            "steps": [
                {"step": "∫(2x+3)dx = x^2 + 3x + C", "explanation": "Integrate term by term", "proof": "Power Rule for Integration: ∫x^n dx = x^(n+1)/(n+1) + C"},
                {"step": "F(2) = 4 + 6 = 10", "explanation": "Evaluate antiderivative at upper bound", "proof": "FTC Part 2"},
                {"step": "F(0) = 0 + 0 = 0", "explanation": "Evaluate at lower bound", "proof": "FTC Part 2"},
                {"step": "10 - 0 = 6", "explanation": "Subtract", "proof": "FTC: ∫_a^b f(x)dx = F(b) - F(a)"}
            ],
            "proofs": [
                {"theorem": "Fundamental Theorem of Calculus Part 2", "statement": "∫_a^b f(x)dx = F(b) - F(a) where F' = f", "applied_to": "definite integral"},
                {"theorem": "Power Rule for Integration", "statement": "∫x^n dx = x^(n+1)/(n+1) + C for n≠-1", "applied_to": "2x term"}
            ]
        },
        {
            "problem": "Solve: x^2 - 5x + 6 = 0",
            "solution": "x = 2 or x = 3",
            "steps": [
                {"step": "Factor: (x-2)(x-3) = 0", "explanation": "Find two numbers that multiply to 6 and sum to -5: -2 and -3", "proof": "Vieta's formulas: for ax^2+bx+c, roots satisfy sum=-b/a, product=c/a"},
                {"step": "x - 2 = 0 → x = 2", "explanation": "Zero Product Property", "proof": "Zero Product Property: if ab=0 then a=0 or b=0"},
                {"step": "x - 3 = 0 → x = 3", "explanation": "Zero Product Property", "proof": "Zero Product Property"}
            ],
            "proofs": [
                {"theorem": "Zero Product Property", "statement": "If a*b = 0 then a=0 or b=0", "applied_to": "factored form"},
                {"theorem": "Vieta's Formulas", "statement": "For ax^2+bx+c=0, sum of roots = -b/a, product = c/a", "applied_to": "x^2-5x+6=0"}
            ]
        },
        {
            "problem": "Find the limit of (x^2 - 1)/(x - 1) as x approaches 1",
            "solution": "2",
            "steps": [
                {"step": "Direct substitution gives 0/0 (indeterminate)", "explanation": "Substitute x=1: (1-1)/(1-1) = 0/0", "proof": "L'Hôpital's Rule applies when limit is 0/0 or ∞/∞"},
                {"step": "Factor numerator: x^2-1 = (x-1)(x+1)", "explanation": "Difference of squares", "proof": "Difference of Squares: a^2-b^2 = (a-b)(a+b)"},
                {"step": "Cancel (x-1): (x+1)", "explanation": "Since x≠1, we can cancel", "proof": "Cancellation: f(x)/f(x) = 1 for f(x)≠0"},
                {"step": "lim(x→1)(x+1) = 2", "explanation": "Direct substitution now works", "proof": "Continuity of polynomial"}
            ],
            "proofs": [
                {"theorem": "Difference of Squares", "statement": "a^2-b^2 = (a-b)(a+b)", "applied_to": "x^2-1"},
                {"theorem": "Direct Substitution Property", "statement": "If f is continuous at a, lim(x→a)f(x) = f(a)", "applied_to": "x+1 at x=1"},
                {"theorem": "L'Hôpital's Rule (alternative)", "statement": "If lim is 0/0, then lim f/g = lim f'/g'", "applied_to": "alternative method"}
            ]
        },
        {
            "problem": "Prove that the sum of two odd integers is even",
            "solution": "The sum of two odd integers is always even",
            "steps": [
                {"step": "Let a = 2k+1, b = 2m+1 for integers k,m", "explanation": "Definition of odd: any odd integer can be written as 2k+1", "proof": "Definition of odd integers"},
                {"step": "a + b = 2k+1 + 2m+1 = 2(k+m) + 2 = 2(k+m+1)", "explanation": "Factor out 2", "proof": "Distributive property and algebraic manipulation"},
                {"step": "2(k+m+1) is even since (k+m+1) is an integer", "explanation": "Any number of form 2n where n is integer is even", "proof": "Definition of even: n is even iff n = 2j for some integer j"}
            ],
            "proofs": [
                {"theorem": "Definition of Odd", "statement": "n is odd iff n = 2k+1 for some integer k", "applied_to": "a and b"},
                {"theorem": "Definition of Even", "statement": "n is even iff n = 2j for some integer j", "applied_to": "a+b"},
                {"theorem": "Closure of Integers under Addition", "statement": "If k,m are integers, k+m+1 is an integer", "applied_to": "k+m+1"}
            ]
        },
        {
            "problem": "Find the eigenvalues of the matrix [[2,1],[1,2]]",
            "solution": "λ = 1 and λ = 3",
            "steps": [
                {"step": "det(A - λI) = det([[2-λ,1],[1,2-λ]]) = 0", "explanation": "Characteristic equation", "proof": "Eigenvalue definition: Av = λv, so det(A-λI)=0"},
                {"step": "(2-λ)^2 - 1 = 0", "explanation": "Expand determinant of 2x2", "proof": "det([[a,b],[c,d]]) = ad-bc"},
                {"step": "λ^2 - 4λ + 3 = 0", "explanation": "Expand and simplify", "proof": "Algebraic expansion"},
                {"step": "(λ-1)(λ-3) = 0", "explanation": "Factor quadratic", "proof": "Factoring"},
                {"step": "λ = 1 or λ = 3", "explanation": "Zero Product Property", "proof": "Zero Product Property"}
            ],
            "proofs": [
                {"theorem": "Characteristic Equation", "statement": "det(A - λI) = 0 gives eigenvalues", "applied_to": "2x2 matrix"},
                {"theorem": "Determinant of 2x2", "statement": "det([[a,b],[c,d]]) = ad-bc", "applied_to": "characteristic matrix"}
            ]
        },
        {
            "problem": "Compute the Taylor series of e^x around x=0 up to degree 3",
            "solution": "1 + x + x^2/2 + x^3/6",
            "steps": [
                {"step": "f(0) = e^0 = 1", "explanation": "Zeroth derivative at 0", "proof": "Taylor: f(0) gives constant term"},
                {"step": "f'(0) = e^0 = 1", "explanation": "First derivative of e^x is e^x", "proof": "d/dx(e^x) = e^x"},
                {"step": "f''(0) = e^0 = 1", "explanation": "All derivatives of e^x are e^x", "proof": "d^n/dx^n(e^x) = e^x for all n"},
                {"step": "f'''(0) = e^0 = 1", "explanation": "Same", "proof": "Exponential derivative property"},
                {"step": "T_3(x) = 1 + x + x^2/2 + x^3/6", "explanation": "Taylor formula: Σ f^(n)(0)/n! * x^n", "proof": "Taylor Series Theorem: f(x) ≈ Σ f^(n)(a)/n! (x-a)^n"}
            ],
            "proofs": [
                {"theorem": "Taylor Series", "statement": "f(x) = Σ f^(n)(a)/n! (x-a)^n", "applied_to": "e^x at a=0"},
                {"theorem": "Derivative of Exponential", "statement": "d^n/dx^n(e^x) = e^x for all n≥0", "applied_to": "all Taylor coefficients"}
            ]
        },
        {
            "problem": "Find the area of a circle with radius 5 using integration",
            "solution": "25π",
            "steps": [
                {"step": "Circle: x^2 + y^2 = 25, so y = ±√(25-x^2)", "explanation": "Solve for y", "proof": "Pythagorean relation on circle"},
                {"step": "A = ∫_{-5}^{5} 2√(25-x^2) dx", "explanation": "Area = 2× upper half (symmetry)", "proof": "Area under curve = integral; symmetry doubles upper half"},
                {"step": "Substitute x = 5sin(θ), dx = 5cos(θ)dθ", "explanation": "Trigonometric substitution", "proof": "Standard trig substitution for √(a^2-x^2)"},
                {"step": "A = 2∫_{-π/2}^{π/2} 25cos^2(θ) dθ = 50∫ cos^2(θ)dθ", "explanation": "Simplify", "proof": "Substitution and algebra"},
                {"step": "Using cos^2(θ) = (1+cos(2θ))/2", "explanation": "Half-angle identity", "proof": "Double angle formula: cos^2 = (1+cos2θ)/2"},
                {"step": "A = 25[θ + sin(2θ)/2]_{-π/2}^{π/2} = 25π", "explanation": "Evaluate", "proof": "Fundamental Theorem of Calculus"}
            ],
            "problems": [
                {"theorem": "Trigonometric Substitution", "statement": "For √(a^2-x^2), use x=a·sin(θ)", "applied_to": "√(25-x^2)"},
                {"theorem": "Half-Angle Identity", "statement": "cos^2(θ) = (1+cos(2θ))/2", "applied_to": "integrating cos^2"},
                {"theorem": "Fundamental Theorem of Calculus", "statement": "∫_a^b f(x)dx = F(b)-F(a)", "applied_to": "definite integral evaluation"}
            ]
        }
    ]

    with open(DATA_FILE, 'w') as f:
        for s in samples:
            messages = [
                {"role": "system", "content": "You are a math expert. Solve problems step by step with proofs for each step."},
                {"role": "user", "content": s["problem"]},
                {"role": "assistant", "content": json.dumps({
                    "answer": s["solution"],
                    "steps": s["steps"],
                    "proofs": s.get("proofs", s.get("problems", [])),
                    "method": "analytical"
                }, indent=2)}
            ]
            f.write(json.dumps({"messages": messages}) + "\n")

    print(f"Generated {len(samples)} training samples at {DATA_FILE}")
    return DATA_FILE


def train():
    """Fine-tune the model with LoRA on math data."""
    generate_training_data()

    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import load_dataset
    import torch

    print(f"\nLoading model {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    # Also load code generation training data if available
    code_data_file = os.path.join(DATA_DIR, "math_code_train.jsonl")
    if os.path.exists(code_data_file):
        code_dataset = load_dataset("json", data_files=code_data_file, split="train")
        from datasets import concatenate_datasets
        dataset = concatenate_datasets([dataset, code_dataset])
        print(f"Combined dataset: {len(dataset)} samples (solution + code generation)")

    def format_and_tokenize(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return tokenizer(text, truncation=True, max_length=1024, padding="max_length")

    dataset = dataset.map(format_and_tokenize, remove_columns=dataset.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    output_dir = os.path.join(TRAINING_DIR, "math-lora")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=2,
        save_steps=50,
        save_total_limit=2,
        warmup_steps=10,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("\n🚀 Starting training...")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✅ Training complete! LoRA adapter saved to {output_dir}")
    print(f"   The agent will automatically use this adapter when available.")


if __name__ == "__main__":
    train()