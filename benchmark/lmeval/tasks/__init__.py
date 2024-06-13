from pprint import pprint
from typing import List, Union
import benchmark.lmeval.base

from . import winogrande
from . import wsc273
from . import webqs
from . import race
from . import anli
from . import squad
from . import lambada
from . import pubmedqa
from . import blimp
from . import gsm8k
from . import openbookqa
from . import arc
from . import sciq
from . import swag
from . import wikitext
from . import piqa
from . import hellaswag
from . import asdiv
from . import coqa
from . import drop
from . import crowspairs
from . import headqa
from . import glue
# from . import hendrycks_ethics
# from . import hendrycks_math
from . import logiqa
from . import mathqa
from . import mc_taco
from . import mutual
# from . import pile
from . import qa4mre
from . import qasper
# from . import sat
from . import toxigen
from . import triviaqa
# from . import unscramble


# task_list = ['winogrande', 'wsc273', 'webqs', 'race', 'anli', 'squad2', 'lambada', 'pubmedqa', 'blimp', 'gsm8k', 'openbookqa', 'arc', 'sciq', 'swag', ]

TASK_REGISTRY = {
    "winogrande": winogrande.Winogrande,
    "wsc273":wsc273.WinogradSchemaChallenge273,
    "webqs":webqs.WebQs,
    "race":race.RACE,
    "anli_r1": anli.ANLIRound1,
    "anli_r2": anli.ANLIRound2,
    "anli_r3": anli.ANLIRound3,
    "squad2": squad.SQuAD2,
    "lambada": lambada.LambadaStandard,
    "pubmedqa": pubmedqa.Pubmed_QA,
    "blimp_adjunct_island": blimp.BlimpAdjunctIsland,
    "blimp_anaphor_gender_agreement": blimp.BlimpAnaphorGenderAgreement,
    "blimp_anaphor_number_agreement": blimp.BlimpAnaphorNumberAgreement,
    "blimp_animate_subject_passive": blimp.BlimpAnimateSubjectPassive,
    "blimp_animate_subject_trans": blimp.BlimpAnimateSubjectTrans,
    "blimp_causative": blimp.BlimpCausative,
    "blimp_complex_NP_island": blimp.BlimpComplex_NPIsland,
    "blimp_coordinate_structure_constraint_complex_left_branch": blimp.BlimpCoordinateStructureConstraintComplexLeftBranch,
    "blimp_coordinate_structure_constraint_object_extraction": blimp.BlimpCoordinateStructureConstraintObjectExtraction,
    "blimp_determiner_noun_agreement_1": blimp.BlimpDeterminerNounAgreement_1,
    "blimp_determiner_noun_agreement_2": blimp.BlimpDeterminerNounAgreement_2,
    "blimp_determiner_noun_agreement_irregular_1": blimp.BlimpDeterminerNounAgreementIrregular_1,
    "blimp_determiner_noun_agreement_irregular_2": blimp.BlimpDeterminerNounAgreementIrregular_2,
    "blimp_determiner_noun_agreement_with_adj_2": blimp.BlimpDeterminerNounAgreementWithAdj_2,
    "blimp_determiner_noun_agreement_with_adj_irregular_1": blimp.BlimpDeterminerNounAgreementWithAdjIrregular_1,
    "blimp_determiner_noun_agreement_with_adj_irregular_2": blimp.BlimpDeterminerNounAgreementWithAdjIrregular_2,
    "blimp_determiner_noun_agreement_with_adjective_1": blimp.BlimpDeterminerNounAgreementWithAdjective_1,
    "blimp_distractor_agreement_relational_noun": blimp.BlimpDistractorAgreementRelationalNoun,
    "blimp_distractor_agreement_relative_clause": blimp.BlimpDistractorAgreementRelativeClause,
    "blimp_drop_argument": blimp.BlimpDropArgument,
    "blimp_ellipsis_n_bar_1": blimp.BlimpEllipsisNBar_1,
    "blimp_ellipsis_n_bar_2": blimp.BlimpEllipsisNBar_2,
    "blimp_existential_there_object_raising": blimp.BlimpExistentialThereObjectRaising,
    "blimp_existential_there_quantifiers_1": blimp.BlimpExistentialThereQuantifiers_1,
    "blimp_existential_there_quantifiers_2": blimp.BlimpExistentialThereQuantifiers_2,
    "blimp_existential_there_subject_raising": blimp.BlimpExistentialThereSubjectRaising,
    "blimp_expletive_it_object_raising": blimp.BlimpExpletiveItObjectRaising,
    "blimp_inchoative": blimp.BlimpInchoative,
    "blimp_intransitive": blimp.BlimpIntransitive,
    "blimp_irregular_past_participle_adjectives": blimp.BlimpIrregularPastParticipleAdjectives,
    "blimp_irregular_past_participle_verbs": blimp.BlimpIrregularPastParticipleVerbs,
    "blimp_irregular_plural_subject_verb_agreement_1": blimp.BlimpIrregularPluralSubjectVerbAgreement_1,
    "blimp_irregular_plural_subject_verb_agreement_2": blimp.BlimpIrregularPluralSubjectVerbAgreement_2,
    "blimp_left_branch_island_echo_question": blimp.BlimpLeftBranchIslandEchoQuestion,
    "blimp_left_branch_island_simple_question": blimp.BlimpLeftBranchIslandSimpleQuestion,
    "blimp_matrix_question_npi_licensor_present": blimp.BlimpMatrixQuestionNpiLicensorPresent,
    "blimp_npi_present_1": blimp.BlimpNpiPresent_1,
    "blimp_npi_present_2": blimp.BlimpNpiPresent_2,
    "blimp_only_npi_licensor_present": blimp.BlimpOnlyNpiLicensorPresent,
    "blimp_only_npi_scope": blimp.BlimpOnlyNpiScope,
    "blimp_passive_1": blimp.BlimpPassive_1,
    "blimp_passive_2": blimp.BlimpPassive_2,
    "blimp_principle_A_c_command": blimp.BlimpPrinciple_ACCommand,
    "blimp_principle_A_case_1": blimp.BlimpPrinciple_ACase_1,
    "blimp_principle_A_case_2": blimp.BlimpPrinciple_ACase_2,
    "blimp_principle_A_domain_1": blimp.BlimpPrinciple_ADomain_1,
    "blimp_principle_A_domain_2": blimp.BlimpPrinciple_ADomain_2,
    "blimp_principle_A_domain_3": blimp.BlimpPrinciple_ADomain_3,
    "blimp_principle_A_reconstruction": blimp.BlimpPrinciple_AReconstruction,
    "blimp_regular_plural_subject_verb_agreement_1": blimp.BlimpRegularPluralSubjectVerbAgreement_1,
    "blimp_regular_plural_subject_verb_agreement_2": blimp.BlimpRegularPluralSubjectVerbAgreement_2,
    "blimp_sentential_negation_npi_licensor_present": blimp.BlimpSententialNegationNpiLicensorPresent,
    "blimp_sentential_negation_npi_scope": blimp.BlimpSententialNegationNpiScope,
    "blimp_sentential_subject_island": blimp.BlimpSententialSubjectIsland,
    "blimp_superlative_quantifiers_1": blimp.BlimpSuperlativeQuantifiers_1,
    "blimp_superlative_quantifiers_2": blimp.BlimpSuperlativeQuantifiers_2,
    "blimp_tough_vs_raising_1": blimp.BlimpToughVsRaising_1,
    "blimp_tough_vs_raising_2": blimp.BlimpToughVsRaising_2,
    "blimp_transitive": blimp.BlimpTransitive,
    "blimp_wh_island": blimp.BlimpWhIsland,
    "blimp_wh_questions_object_gap": blimp.BlimpWhQuestionsObjectGap,
    "blimp_wh_questions_subject_gap": blimp.BlimpWhQuestionsSubjectGap,
    "blimp_wh_questions_subject_gap_long_distance": blimp.BlimpWhQuestionsSubjectGapLongDistance,
    "blimp_wh_vs_that_no_gap": blimp.BlimpWhVsThatNoGap,
    "blimp_wh_vs_that_no_gap_long_distance": blimp.BlimpWhVsThatNoGapLongDistance,
    "blimp_wh_vs_that_with_gap": blimp.BlimpWhVsThatWithGap,
    "blimp_wh_vs_that_with_gap_long_distance": blimp.BlimpWhVsThatWithGapLongDistance,
    "gsm8k": gsm8k.GradeSchoolMath8K,
    "openbookqa": openbookqa.OpenBookQA,
    "arc_easy": arc.ARCEasy,
    "arc_challenge": arc.ARCChallenge,
    "sciq": sciq.SciQ,
    "swag": swag.SWAG,
    "wikitext": wikitext.WikiText,
    "piqa": piqa.PiQA,
    "hellaswag": hellaswag.HellaSwag,
    "math_asdiv": asdiv.Asdiv,
    "coqa": coqa.CoQA,
    "drop": drop.DROP,
    "crows_pairs_english": crowspairs.CrowsPairsEnglish,
    "crows_pairs_english_race_color": crowspairs.CrowsPairsEnglishRaceColor,
    "crows_pairs_english_socioeconomic": crowspairs.CrowsPairsEnglishSocioeconomic,
    "crows_pairs_english_gender": crowspairs.CrowsPairsEnglishGender,
    "crows_pairs_english_age": crowspairs.CrowsPairsEnglishAge,
    "crows_pairs_english_religion": crowspairs.CrowsPairsEnglishReligion,
    "crows_pairs_english_disability": crowspairs.CrowsPairsEnglishDisability,
    "crows_pairs_english_sexual_orientation": crowspairs.CrowsPairsEnglishSexualOrientation,
    "crows_pairs_english_nationality": crowspairs.CrowsPairsEnglishNationality,
    "crows_pairs_english_physical_appearance": crowspairs.CrowsPairsEnglishPhysicalAppearance,
    "crows_pairs_english_autre": crowspairs.CrowsPairsEnglishAutre,
    "crows_pairs_french": crowspairs.CrowsPairsFrench,
    "crows_pairs_french_race_color": crowspairs.CrowsPairsFrenchRaceColor,
    "crows_pairs_french_socioeconomic": crowspairs.CrowsPairsFrenchSocioeconomic,
    "crows_pairs_french_gender": crowspairs.CrowsPairsFrenchGender,
    "crows_pairs_french_age": crowspairs.CrowsPairsFrenchAge,
    "crows_pairs_french_religion": crowspairs.CrowsPairsFrenchReligion,
    "crows_pairs_french_disability": crowspairs.CrowsPairsFrenchDisability,
    "crows_pairs_french_sexual_orientation": crowspairs.CrowsPairsFrenchSexualOrientation,
    "crows_pairs_french_nationality": crowspairs.CrowsPairsFrenchNationality,
    "crows_pairs_french_physical_appearance": crowspairs.CrowsPairsFrenchPhysicalAppearance,
    "crows_pairs_french_autre": crowspairs.CrowsPairsFrenchAutre,
    "headqa": headqa.HeadQAEsDeprecated, 
    "headqa_es": headqa.HeadQAEs,
    "headqa_en": headqa.HeadQAEn,
    # "glue_cola": glue.CoLA,
    "glue_mnli": glue.MNLI,
    "glue_mnli_mismatched": glue.MNLIMismatched,
    "glue_mrpc": glue.MRPC,
    "glue_rte": glue.RTE,
    "glue_qnli": glue.QNLI,
    "glue_qqp": glue.QQP,
    "glue_sst": glue.SST,
    "glue_wnli": glue.WNLI,
    # "ethics_cm": hendrycks_ethics.EthicsCM,
    # "ethics_deontology": hendrycks_ethics.EthicsDeontology,
    # "ethics_justice": hendrycks_ethics.EthicsJustice,
    # "ethics_utilitarianism_original": hendrycks_ethics.EthicsUtilitarianismOriginal,
    # "ethics_utilitarianism": hendrycks_ethics.EthicsUtilitarianism,
    # "ethics_virtue": hendrycks_ethics.EthicsVirtue,
    # "math_algebra": hendrycks_math.MathAlgebra,
    # "math_counting_and_prob": hendrycks_math.MathCountingAndProbability,
    # "math_geometry": hendrycks_math.MathGeometry,
    # "math_intermediate_algebra": hendrycks_math.MathIntermediateAlgebra,
    # "math_num_theory": hendrycks_math.MathNumberTheory,
    # "math_prealgebra": hendrycks_math.MathPrealgebra,
    # "math_precalc": hendrycks_math.MathPrecalculus,
    "logiqa": logiqa.LogiQA,
    "mathqa": mathqa.MathQA,
    "mc_taco": mc_taco.MCTACO,
    "mutual": mutual.MuTual,
    "mutual_plus": mutual.MuTualPlus,
    # "pile_arxiv": pile.PileArxiv,
    # "pile_books3": pile.PileBooks3,
    # "pile_bookcorpus2": pile.PileBookCorpus2,
    # "pile_dm-mathematics": pile.PileDmMathematics,
    # "pile_enron": pile.PileEnron,
    # "pile_europarl": pile.PileEuroparl,
    # "pile_freelaw": pile.PileFreeLaw,
    # "pile_github": pile.PileGithub,
    # "pile_gutenberg": pile.PileGutenberg,
    # "pile_hackernews": pile.PileHackernews,
    # "pile_nih-exporter": pile.PileNIHExporter,
    # "pile_opensubtitles": pile.PileOpenSubtitles,
    # "pile_openwebtext2": pile.PileOpenWebText2,
    # "pile_philpapers": pile.PilePhilPapers,
    # "pile_pile-cc": pile.PilePileCc,
    # "pile_pubmed-abstracts": pile.PilePubmedAbstracts,
    # "pile_pubmed-central": pile.PilePubmedCentral,
    # "pile_stackexchange": pile.PileStackExchange,
    # "pile_uspto": pile.PileUspto,
    # "pile_ubuntu-irc": pile.PileUbuntuIrc,
    # "pile_wikipedia": pile.PileWikipedia,
    # "pile_youtubesubtitles": pile.PileYoutubeSubtitles,
    "qa4mre_2011": qa4mre.QA4MRE_2011,
    "qa4mre_2012": qa4mre.QA4MRE_2012,
    "qa4mre_2013": qa4mre.QA4MRE_2013,
    "qasper": qasper.QASPER,
    # "sat": sat.SATAnalogies,
    "toxigen": toxigen.ToxiGen,
    "triviaqa": triviaqa.TriviaQA,
    # "anagrams1": unscramble.Anagrams1,
    # "anagrams2": unscramble.Anagrams2,
    # "cycle_letters": unscramble.CycleLetters,
    # "random_insertion": unscramble.RandomInsertion,
    # "reversed_words": unscramble.ReversedWords,


}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")


def get_task_name_from_object(task_object):
    for name, class_ in TASK_REGISTRY.items():
        if class_ is task_object:
            return name

    # this gives a mechanism for non-registered tasks to have a custom name anyways when reporting
    return (
        task_object.EVAL_HARNESS_NAME
        if hasattr(task_object, "EVAL_HARNESS_NAME")
        else type(task_object).__name__
    )


def get_task_dict(task_name_list: List[Union[str, benchmark.lmeval.base.Task]]):
    task_name_dict = {
        task_name: get_task(task_name)()
        for task_name in task_name_list
        if isinstance(task_name, str)
    }
    task_name_from_object_dict = {
        get_task_name_from_object(task_object): task_object
        for task_object in task_name_list
        if not isinstance(task_object, str)
    }
    assert set(task_name_dict.keys()).isdisjoint(set(task_name_from_object_dict.keys()))
    return {**task_name_dict, **task_name_from_object_dict}
