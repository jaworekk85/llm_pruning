from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


PROMPT_TYPES = ["definition", "factual", "explanation", "comparison", "application"]
SPLIT_BY_CONCEPT_INDEX = {
    "discovery": range(0, 6),
    "validation": range(6, 8),
    "test": range(8, 10),
}
VARIANTS_PER_TYPE_BY_PROFILE = {
    "pilot": {
        "discovery": 1,
        "validation": 1,
        "test": 1,
    },
    "large": {
        "discovery": 5,
        "validation": 10,
        "test": 10,
    },
}


@dataclass(frozen=True)
class Concept:
    term: str
    definition: str
    feature: str
    importance: str
    related_term: str
    contrast: str
    application: str
    difficulty: str


BLUEPRINTS: dict[str, list[Concept]] = {
    "agriculture": [
        Concept("crop rotation", "the practice of growing different crops on the same land across seasons", "it changes crop families and nutrient demands over time", "it can reduce pests, improve soil structure, and balance nutrient use", "monoculture", "monoculture repeats one crop, while crop rotation deliberately changes crops over time", "a farmer can rotate legumes with cereals to add nitrogen and reduce disease pressure", "basic"),
        Concept("irrigation", "the controlled supply of water to crops", "it supplements rainfall when natural moisture is insufficient", "it helps plants maintain growth during dry periods", "rainfed farming", "rainfed farming depends mainly on rainfall, while irrigation adds managed water", "a grower can schedule irrigation when soil moisture drops below crop needs", "basic"),
        Concept("soil fertility", "the ability of soil to supply nutrients that plants need", "it depends on nutrients, organic matter, pH, and biological activity", "it strongly affects crop yield and long-term farm productivity", "soil texture", "soil texture describes particle size, while soil fertility describes nutrient-supplying capacity", "a farmer can test soil and add compost or fertilizer to correct nutrient shortages", "basic"),
        Concept("integrated pest management", "a strategy that combines biological, cultural, mechanical, and chemical pest controls", "it uses monitoring before choosing a control method", "it can reduce crop losses while limiting unnecessary pesticide use", "routine spraying", "routine spraying applies pesticides by schedule, while integrated pest management responds to observed pest risk", "a grower can scout fields and release beneficial insects before using a targeted pesticide", "intermediate"),
        Concept("cover crops", "plants grown mainly to protect or improve soil rather than to harvest", "they can cover bare soil between cash crops", "they reduce erosion, add organic matter, and can suppress weeds", "cash crops", "cash crops are grown for sale, while cover crops are grown mainly for soil benefits", "a farmer can plant rye after harvest to protect soil through winter", "basic"),
        Concept("greenhouse cultivation", "growing plants inside a controlled structure", "it allows control over temperature, humidity, and light exposure", "it can extend growing seasons and protect crops from harsh weather", "open-field cultivation", "open-field cultivation is exposed to weather, while greenhouse cultivation modifies the crop environment", "a producer can grow tomatoes earlier in spring by controlling greenhouse temperature", "intermediate"),
        Concept("livestock grazing", "the feeding of animals on pasture or rangeland", "it converts forage into meat, milk, wool, or labor", "it can support food production but must be managed to avoid overgrazing", "feedlot feeding", "feedlot feeding brings feed to confined animals, while grazing moves animals across forage land", "a rancher can rotate animals between paddocks so grass can recover", "basic"),
        Concept("precision agriculture", "the use of sensors, maps, and data to manage fields more precisely", "it applies inputs according to spatial variation within a field", "it can reduce waste and improve efficiency in fertilizer, water, and pesticide use", "uniform application", "uniform application treats the whole field the same, while precision agriculture varies treatment by location", "a farmer can use yield maps to apply more fertilizer only where it is needed", "advanced"),
        Concept("compost", "decomposed organic material used to improve soil", "it contains stabilized plant or animal residues", "it adds organic matter and can improve water retention and nutrient cycling", "synthetic fertilizer", "synthetic fertilizer supplies concentrated nutrients, while compost also adds organic matter", "a grower can mix compost into beds before planting vegetables", "basic"),
        Concept("crop yield", "the amount of crop harvested per unit of land", "it is often measured in kilograms or tons per hectare", "it summarizes agricultural productivity and helps compare management choices", "crop quality", "crop yield measures quantity, while crop quality measures traits such as size, flavor, or protein content", "a farmer can compare yields from two varieties to decide which performs better", "basic"),
    ],
    "astronomy": [
        Concept("star", "a massive sphere of hot plasma that produces energy through nuclear fusion", "it emits light and heat from its surface", "stars are the main visible building blocks of galaxies", "planet", "a planet does not produce fusion energy, while a star does", "an observer can classify a star by its color, brightness, and spectrum", "basic"),
        Concept("planetary orbit", "the curved path a planet follows around a star", "it results from gravity and the planet's forward motion", "orbits explain seasons, years, and many predictable sky patterns", "straight-line motion", "straight-line motion has no central gravitational curve, while an orbit is continuously bent by gravity", "an astronomer can use orbital period to estimate a planet's distance from its star", "basic"),
        Concept("solar eclipse", "an event in which the Moon blocks sunlight from reaching part of Earth", "it occurs when the Moon passes between Earth and the Sun", "it reveals how alignment and shadows work in the Earth-Moon-Sun system", "lunar eclipse", "a solar eclipse involves the Moon blocking the Sun, while a lunar eclipse involves Earth shadowing the Moon", "a teacher can use eclipse paths to show why alignment must be precise", "basic"),
        Concept("galaxy", "a large gravitationally bound system of stars, gas, dust, and dark matter", "it can contain billions of stars", "galaxies organize much of the visible matter in the universe", "solar system", "a solar system surrounds one star, while a galaxy contains many stars and systems", "an astronomer can study galaxy shape to infer its history", "basic"),
        Concept("telescope", "an instrument that collects radiation from distant objects", "it increases light-gathering power and angular detail", "telescopes let astronomers observe objects too faint or distant for unaided eyes", "microscope", "a microscope magnifies small nearby objects, while a telescope observes distant objects", "a student can use a telescope to observe lunar craters", "basic"),
        Concept("nebula", "a cloud of gas and dust in space", "some nebulae are regions where stars form", "nebulae reveal processes of star birth, death, and chemical enrichment", "star cluster", "a nebula is gas and dust, while a star cluster is a group of stars", "an astronomer can image a nebula to study active star formation", "intermediate"),
        Concept("black hole", "a region where gravity is so strong that nothing inside the event horizon can escape", "it can form from the collapse of a massive star", "black holes test theories of gravity and influence nearby matter", "neutron star", "a neutron star has a surface, while a black hole has an event horizon", "an astronomer can infer a black hole from hot gas orbiting an unseen compact object", "advanced"),
        Concept("exoplanet", "a planet that orbits a star outside the Solar System", "it can be detected by transits or stellar wobble", "exoplanets help scientists study how common planetary systems are", "moon", "a moon orbits a planet, while an exoplanet orbits a star beyond the Solar System", "a telescope can detect an exoplanet when it slightly dims its star during transit", "intermediate"),
        Concept("light-year", "the distance light travels in one year", "it is a unit of distance, not time", "it helps describe enormous astronomical distances", "astronomical unit", "an astronomical unit is based on Earth-Sun distance, while a light-year is much larger", "an astronomer can say a nearby star is several light-years away", "basic"),
        Concept("cosmic microwave background", "faint microwave radiation left from the early universe", "it is observed in nearly every direction in space", "it provides evidence about the hot early state of the universe", "starlight", "starlight comes from stars, while the cosmic microwave background comes from the early universe", "cosmologists can map tiny temperature variations to study early structure", "advanced"),
    ],
    "math": [
        Concept("prime number", "a whole number greater than one with exactly two positive divisors", "it is divisible only by one and itself", "prime numbers are fundamental building blocks of multiplication", "composite number", "a composite number has additional divisors, while a prime number has exactly two", "a student can test divisibility to decide whether 29 is prime", "basic"),
        Concept("linear equation", "an equation whose graph is a straight line", "the variable appears only to the first power", "linear equations model constant-rate relationships", "quadratic equation", "a quadratic equation includes a squared term, while a linear equation does not", "a student can isolate the variable to solve a one-step linear equation", "basic"),
        Concept("Pythagorean theorem", "a relationship among the side lengths of a right triangle", "it states that the hypotenuse squared equals the sum of the squares of the legs", "it is used to compute distances and missing side lengths", "triangle inequality", "the triangle inequality limits possible side lengths, while the Pythagorean theorem applies specifically to right triangles", "a student can find a missing hypotenuse using a squared plus b squared equals c squared", "basic"),
        Concept("derivative", "a measure of instantaneous rate of change", "it gives the slope of a function at a point", "derivatives are central to optimization and motion problems", "average rate of change", "average rate uses an interval, while a derivative describes an instantaneous rate", "a student can use a derivative to find where a function is increasing fastest", "intermediate"),
        Concept("probability", "a measure of how likely an event is to occur", "it ranges from zero to one in standard notation", "probability supports reasoning under uncertainty", "frequency", "frequency counts observed occurrences, while probability describes expected likelihood", "a student can compute the probability of rolling a six on a fair die", "basic"),
        Concept("matrix", "a rectangular array of numbers or symbols", "it has rows and columns", "matrices represent linear transformations and systems of equations", "vector", "a vector is one-dimensional, while a matrix has rows and columns", "a student can use a matrix to organize coefficients in a system of equations", "intermediate"),
        Concept("integral", "a mathematical object that accumulates quantities over an interval", "it can represent area under a curve", "integrals are used in geometry, physics, probability, and accumulation problems", "derivative", "a derivative measures change, while an integral accumulates change", "a student can use an integral to find distance from a velocity function", "intermediate"),
        Concept("function", "a rule that assigns each input exactly one output", "it connects variables in a structured way", "functions provide a language for modeling relationships", "relation", "a relation can pair one input with many outputs, while a function assigns exactly one output", "a student can evaluate a function by substituting a value for the input", "basic"),
        Concept("proof", "a logical argument that establishes why a mathematical statement is true", "it relies on definitions, assumptions, and valid reasoning", "proofs distinguish demonstrated truth from numerical pattern spotting", "example", "an example illustrates a case, while a proof covers all cases under its assumptions", "a student can prove an even number plus an even number is even using algebra", "intermediate"),
        Concept("standard deviation", "a measure of how spread out data values are around their mean", "larger values indicate more variability", "it helps compare consistency and variation across datasets", "mean", "the mean describes central value, while standard deviation describes spread", "a student can compare two test-score groups by their standard deviations", "basic"),
    ],
    "medicine": [
        Concept("vaccine", "a preparation that trains the immune system to recognize a pathogen", "it can contain weakened, inactive, partial, or encoded pathogen information", "vaccines help prevent infectious disease and reduce severe illness", "antibiotic", "an antibiotic treats bacterial infection, while a vaccine prepares immunity before exposure", "a clinician can recommend vaccination to reduce risk of a preventable disease", "basic"),
        Concept("blood pressure", "the force of blood pushing against artery walls", "it is recorded as systolic over diastolic pressure", "blood pressure helps assess cardiovascular strain and health risk", "heart rate", "heart rate counts beats per minute, while blood pressure measures force against vessels", "a nurse can measure blood pressure to screen for hypertension", "basic"),
        Concept("antibiotic", "a medicine used to treat bacterial infections", "it kills bacteria or slows bacterial growth", "antibiotics are essential for treating many bacterial diseases", "antiviral", "an antiviral targets viruses, while an antibiotic targets bacteria", "a doctor can prescribe an antibiotic when evidence suggests bacterial infection", "basic"),
        Concept("immune system", "the body's defense network against harmful microbes and abnormal cells", "it includes cells, tissues, signaling molecules, and antibodies", "it protects the body from infection and supports healing", "nervous system", "the nervous system sends signals, while the immune system detects and responds to threats", "a clinician can explain fever as part of an immune response", "basic"),
        Concept("dehydration", "a state in which the body has too little water for normal function", "it can involve loss of water and electrolytes", "dehydration can impair circulation, temperature control, and organ function", "malnutrition", "malnutrition concerns nutrient imbalance, while dehydration concerns fluid deficiency", "a caregiver can offer oral rehydration solution after vomiting or diarrhea", "basic"),
        Concept("inflammation", "a protective biological response to injury, infection, or irritation", "it can cause redness, heat, swelling, pain, and loss of function", "inflammation helps isolate harm and begin repair, but chronic inflammation can damage tissue", "infection", "infection means invasion by microbes, while inflammation is the body's response and can occur without infection", "a doctor can evaluate swelling to decide whether inflammation is acute or chronic", "intermediate"),
        Concept("diabetes", "a condition involving problems with blood glucose regulation", "it can result from insufficient insulin or reduced insulin response", "diabetes management reduces risks to eyes, kidneys, nerves, and blood vessels", "hypoglycemia", "diabetes is a chronic regulation problem, while hypoglycemia is a low blood sugar state", "a patient can monitor blood glucose to guide diet, medication, and activity choices", "intermediate"),
        Concept("anemia", "a condition with too few red blood cells or too little hemoglobin", "it reduces the blood's ability to carry oxygen", "anemia can cause fatigue, weakness, and shortness of breath", "hypoxia", "hypoxia means low tissue oxygen, while anemia is one possible cause involving blood oxygen-carrying capacity", "a clinician can order a blood count to evaluate suspected anemia", "intermediate"),
        Concept("diagnosis", "the process of identifying a disease or condition", "it combines history, examination, tests, and clinical reasoning", "diagnosis guides treatment and helps estimate prognosis", "screening", "screening looks for possible disease in at-risk people, while diagnosis tries to identify a specific condition", "a doctor can combine symptoms and test results to make a diagnosis", "basic"),
        Concept("clinical trial", "a structured study that tests a medical intervention in people", "it follows a protocol and often compares groups", "clinical trials provide evidence about safety and effectiveness", "case report", "a case report describes an individual patient, while a clinical trial systematically tests an intervention", "researchers can randomize participants to compare a new treatment with standard care", "advanced"),
    ],
    "politics": [
        Concept("constitution", "a fundamental set of rules for organizing government", "it defines institutions, powers, limits, and rights", "constitutions provide legal structure and constrain political authority", "ordinary law", "ordinary law is made under the constitution, while the constitution sets the higher framework", "citizens can use constitutional rights to challenge unlawful government action", "basic"),
        Concept("election", "a process for choosing representatives or deciding public questions by vote", "it uses rules about eligibility, ballots, counting, and outcomes", "elections are a core mechanism of democratic accountability", "appointment", "appointment selects officials without a public vote, while an election uses voting", "voters can compare candidates and cast ballots to choose a representative", "basic"),
        Concept("parliament", "a representative body that debates and makes laws", "it often approves budgets and scrutinizes government", "parliaments connect public representation with lawmaking", "cabinet", "a cabinet usually leads executive policy, while a parliament debates and passes laws", "a parliament can question ministers about public spending", "basic"),
        Concept("political party", "an organization that groups people around policy goals and electoral competition", "it recruits candidates and organizes campaigns", "parties help structure voter choice and government formation", "interest group", "an interest group advocates on issues, while a political party seeks governing power through elections", "a party can publish a platform before an election", "basic"),
        Concept("separation of powers", "the division of government authority among different branches", "it often separates legislative, executive, and judicial functions", "it can prevent concentration and abuse of power", "fusion of powers", "fusion of powers combines functions, while separation of powers divides them", "a court can review executive action under a separated system", "intermediate"),
        Concept("federalism", "a system that divides authority between central and regional governments", "it assigns powers to more than one level of government", "federalism can balance national unity with regional autonomy", "unitary state", "a unitary state centralizes authority, while federalism constitutionally divides authority", "a regional government can manage education while the national government manages defense", "intermediate"),
        Concept("public policy", "a course of government action addressing public problems", "it can include laws, regulations, spending, and programs", "public policy shapes how governments respond to social needs", "political ideology", "ideology is a set of beliefs, while public policy is a concrete government action", "officials can design a policy to reduce traffic congestion", "basic"),
        Concept("judicial review", "the power of courts to assess whether laws or actions comply with higher law", "it often involves constitutional interpretation", "judicial review can limit unlawful government action", "legislative debate", "legislative debate discusses proposed laws, while judicial review evaluates legality", "a court can strike down a law that violates constitutional protections", "advanced"),
        Concept("civil rights", "legal protections that guarantee equal treatment and participation", "they often protect voting, due process, and freedom from discrimination", "civil rights help define fair membership in a political community", "civil liberties", "civil liberties limit government interference, while civil rights emphasize equal protection and participation", "citizens can invoke civil rights laws to challenge discrimination", "intermediate"),
        Concept("coalition government", "a government formed by multiple political parties working together", "it often occurs when no single party wins a majority", "coalitions allow parliamentary systems to form governing majorities", "single-party government", "a single-party government is controlled by one party, while a coalition shares power among parties", "parties can negotiate a coalition agreement after an election", "intermediate"),
    ],
}


def split_for_concept(index: int) -> str:
    for split, indices in SPLIT_BY_CONCEPT_INDEX.items():
        if index in indices:
            return split
    raise ValueError(f"No split configured for concept index {index}")


def capitalized(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


def definition_variants(domain: str, concept: Concept) -> list[tuple[str, str]]:
    del domain
    return [
        (
            f"What is {concept.term}?",
            f"{capitalized(concept.term)} is {concept.definition}.",
        ),
        (
            f"What does {concept.term} mean?",
            f"{capitalized(concept.term)} means {concept.definition}.",
        ),
        (
            f"How would you define {concept.term}?",
            f"{capitalized(concept.term)} can be defined as {concept.definition}.",
        ),
        (
            f"What is the basic idea of {concept.term}?",
            f"The basic idea of {concept.term} is {concept.definition}.",
        ),
        (
            f"What is a short definition of {concept.term}?",
            f"A short definition of {concept.term} is {concept.definition}.",
        ),
        (
            f"What concept does the term {concept.term} describe?",
            f"The term {concept.term} describes {concept.definition}.",
        ),
        (
            f"In simple terms, what is {concept.term}?",
            f"In simple terms, {concept.term} is {concept.definition}.",
        ),
        (
            f"What does someone mean by {concept.term}?",
            f"Someone who refers to {concept.term} means {concept.definition}.",
        ),
        (
            f"What is meant by {concept.term}?",
            f"What is meant by {concept.term} is {concept.definition}.",
        ),
        (
            f"How can {concept.term} be described?",
            f"{capitalized(concept.term)} can be described as {concept.definition}.",
        ),
    ]


def factual_variants(domain: str, concept: Concept) -> list[tuple[str, str]]:
    del domain
    return [
        (
            f"What is one important feature of {concept.term}?",
            f"One important feature of {concept.term} is that {concept.feature}.",
        ),
        (
            f"What feature helps identify {concept.term}?",
            f"A feature that helps identify {concept.term} is that {concept.feature}.",
        ),
        (
            f"What is a key property of {concept.term}?",
            f"A key property of {concept.term} is that {concept.feature}.",
        ),
        (
            f"What should someone remember about {concept.term}?",
            f"Someone should remember that {concept.term} has this feature: {concept.feature}.",
        ),
        (
            f"What fact is central to {concept.term}?",
            f"A central fact about {concept.term} is that {concept.feature}.",
        ),
        (
            f"What characteristic is associated with {concept.term}?",
            f"A characteristic associated with {concept.term} is that {concept.feature}.",
        ),
        (
            f"What is one trait of {concept.term}?",
            f"One trait of {concept.term} is that {concept.feature}.",
        ),
        (
            f"What makes {concept.term} recognizable?",
            f"{capitalized(concept.term)} is recognizable because {concept.feature}.",
        ),
        (
            f"What detail matters for understanding {concept.term}?",
            f"A detail that matters for understanding {concept.term} is that {concept.feature}.",
        ),
        (
            f"What is a useful fact about {concept.term}?",
            f"A useful fact about {concept.term} is that {concept.feature}.",
        ),
    ]


def explanation_variants(domain: str, concept: Concept) -> list[tuple[str, str]]:
    return [
        (
            f"Why is {concept.term} important in {domain}?",
            f"{capitalized(concept.term)} is important in {domain} because {concept.importance}.",
        ),
        (
            f"Why does {concept.term} matter in {domain}?",
            f"{capitalized(concept.term)} matters in {domain} because {concept.importance}.",
        ),
        (
            f"What makes {concept.term} useful in {domain}?",
            f"{capitalized(concept.term)} is useful in {domain} because {concept.importance}.",
        ),
        (
            f"Why should someone studying {domain} know about {concept.term}?",
            f"Someone studying {domain} should know about {concept.term} because {concept.importance}.",
        ),
        (
            f"How does {concept.term} support work in {domain}?",
            f"{capitalized(concept.term)} supports work in {domain} because {concept.importance}.",
        ),
        (
            f"What role does {concept.term} play in {domain}?",
            f"The role of {concept.term} in {domain} is important because {concept.importance}.",
        ),
        (
            f"Why is {concept.term} a significant idea in {domain}?",
            f"{capitalized(concept.term)} is significant in {domain} because {concept.importance}.",
        ),
        (
            f"How does {concept.term} help explain problems in {domain}?",
            f"{capitalized(concept.term)} helps explain problems in {domain} because {concept.importance}.",
        ),
        (
            f"What benefit does understanding {concept.term} provide in {domain}?",
            f"Understanding {concept.term} helps in {domain} because {concept.importance}.",
        ),
        (
            f"Why might {concept.term} be taught in a {domain} lesson?",
            f"{capitalized(concept.term)} might be taught in a {domain} lesson because {concept.importance}.",
        ),
    ]


def comparison_variants(domain: str, concept: Concept) -> list[tuple[str, str]]:
    del domain
    contrast = capitalized(concept.contrast)
    return [
        (
            f"How is {concept.term} different from {concept.related_term}?",
            f"{contrast}.",
        ),
        (
            f"What is the difference between {concept.term} and {concept.related_term}?",
            f"The difference is that {concept.contrast}.",
        ),
        (
            f"How would you contrast {concept.term} with {concept.related_term}?",
            f"To contrast them, {concept.contrast}.",
        ),
        (
            f"Why is {concept.term} not the same as {concept.related_term}?",
            f"{capitalized(concept.term)} is not the same as {concept.related_term} because {concept.contrast}.",
        ),
        (
            f"What separates {concept.term} from {concept.related_term}?",
            f"What separates them is that {concept.contrast}.",
        ),
        (
            f"How can someone tell {concept.term} apart from {concept.related_term}?",
            f"Someone can tell them apart because {concept.contrast}.",
        ),
        (
            f"What distinction exists between {concept.term} and {concept.related_term}?",
            f"The distinction is that {concept.contrast}.",
        ),
        (
            f"How do {concept.term} and {concept.related_term} differ?",
            f"They differ because {concept.contrast}.",
        ),
        (
            f"What comparison helps explain {concept.term} versus {concept.related_term}?",
            f"A helpful comparison is that {concept.contrast}.",
        ),
        (
            f"In what way is {concept.term} unlike {concept.related_term}?",
            f"{capitalized(concept.term)} is unlike {concept.related_term} because {concept.contrast}.",
        ),
    ]


def application_variants(domain: str, concept: Concept) -> list[tuple[str, str]]:
    application = capitalized(concept.application)
    return [
        (
            f"How could someone apply {concept.term} in a simple {domain} task?",
            f"{application}.",
        ),
        (
            f"What is a simple use of {concept.term} in {domain}?",
            f"A simple use is this: {concept.application}.",
        ),
        (
            f"What is an example of applying {concept.term} in {domain}?",
            f"An example is that {concept.application}.",
        ),
        (
            f"How might a beginner use {concept.term} in {domain}?",
            f"A beginner might use it this way: {concept.application}.",
        ),
        (
            f"What practical task could involve {concept.term} in {domain}?",
            f"A practical task could involve this: {concept.application}.",
        ),
        (
            f"How can {concept.term} be used in practice in {domain}?",
            f"In practice, {concept.application}.",
        ),
        (
            f"What scenario shows {concept.term} being used in {domain}?",
            f"One scenario is that {concept.application}.",
        ),
        (
            f"How could someone demonstrate {concept.term} in {domain}?",
            f"Someone could demonstrate it as follows: {concept.application}.",
        ),
        (
            f"What would an applied example of {concept.term} look like in {domain}?",
            f"An applied example would be this: {concept.application}.",
        ),
        (
            f"How might {concept.term} guide a simple decision in {domain}?",
            f"{capitalized(concept.term)} might guide a decision because {concept.application}.",
        ),
    ]


VARIANT_BUILDERS = {
    "definition": definition_variants,
    "factual": factual_variants,
    "explanation": explanation_variants,
    "comparison": comparison_variants,
    "application": application_variants,
}


def records_for_concept(
    domain: str,
    concept_index: int,
    concept: Concept,
    variants_per_type: int,
) -> list[dict[str, str]]:
    split = split_for_concept(concept_index)
    base_id = f"{domain}.qa_v1.{concept_index:02d}"
    base = {
        "domain": domain,
        "split": split,
        "source_type": "synthetic_from_manual_blueprint",
        "source_name": "qa_v1_blueprints",
        "difficulty": concept.difficulty,
        "language": "en",
        "license": "project",
        "notes": "Generated deterministically by experiments/build_qa_v1.py from manual concept blueprints.",
    }

    rows: list[dict[str, str]] = []
    for prompt_type in PROMPT_TYPES:
        variants = VARIANT_BUILDERS[prompt_type](domain, concept)
        if len(variants) < variants_per_type:
            raise ValueError(
                f"Only {len(variants)} {prompt_type} variants are available, "
                f"but {variants_per_type} were requested."
            )
        for variant_index, (prompt, target) in enumerate(variants[:variants_per_type]):
            record_id = f"{base_id}.{prompt_type}"
            if variants_per_type > 1:
                record_id = f"{record_id}.{variant_index:02d}"
            rows.append(
                {
                    **base,
                    "id": record_id,
                    "prompt_type": prompt_type,
                    "prompt": prompt,
                    "target": target,
                }
            )
    return rows


def build_records(profile: str) -> list[dict[str, str]]:
    variants_by_split = VARIANTS_PER_TYPE_BY_PROFILE[profile]
    records: list[dict[str, str]] = []
    for domain, concepts in BLUEPRINTS.items():
        if len(concepts) != 10:
            raise ValueError(f"Expected 10 concepts for {domain}, got {len(concepts)}")

        for concept_index, concept in enumerate(concepts):
            split = split_for_concept(concept_index)
            records.extend(
                records_for_concept(
                    domain,
                    concept_index,
                    concept,
                    variants_per_type=variants_by_split[split],
                )
            )

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the deterministic QA v1 prompt set.")
    parser.add_argument("--output", type=Path, default=Path("data/prompt_sets/qa_v1.jsonl"))
    parser.add_argument(
        "--profile",
        choices=sorted(VARIANTS_PER_TYPE_BY_PROFILE),
        default="pilot",
        help=(
            "pilot reproduces the original 250-record dataset. "
            "large writes 1,750 records: 150 discovery, 100 validation, "
            "and 100 test records per domain."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_records(args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            file.write("\n")

    print(f"Wrote {len(records)} QA records to {args.output}")


if __name__ == "__main__":
    main()
