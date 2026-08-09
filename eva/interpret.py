"""Plain-language results report relating behavioral scores to EEG activity.

Generates a self-contained HTML summary intended for readers without a
signal-processing background (e.g., clinicians, students), built from a
.h5 file already produced by :func:`eva.preprocess` and :func:`eva.sync`.

Usage
-----
>>> from eva import interpret
>>> interpret("subject01.h5")

>>> # Custom score key and frequency bands
>>> interpret("subject01.h5", score_key="accuracy",
...           bands={"theta": (4.0, 8.0), "alpha": (8.0, 13.0)})

>>> # Human-readable condition names
>>> interpret("subject01.h5", label_names={"vr_att": "Attention",
...                                          "vr_abs": "Abstraction"})

>>> # Portuguese (PT-BR) report, mirroring VECA-EEG's own bilingual support
>>> interpret("subject01.h5", language="pt-br")

Design note
-----------
All values shown are normalised (0-100) relative to the epochs within the
*same session* -- there is no population norm or clinical reference built
into this module. The report is descriptive, not inferential: it does not
compute or display correlation coefficients or p-values, since a single
session provides no statistical power for such claims. It also does not
imply that a taller "activity" bar is uniformly better or worse: the
literature commonly associates different directions with different bands
(e.g. higher theta power with effortful engagement, lower alpha/beta power
with active engagement) -- see the report's own disclaimer text.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .metrics import compute_psd

logger = logging.getLogger(__name__)

# Default bands mirror the ones used in the VECA-EEG pilot analysis.
_DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
}

_SUPPORTED_LANGUAGES = ("en", "pt-br")
_HTML_LANG_ATTR = {"en": "en", "pt-br": "pt-BR"}

# Localized display name per band, keyed by lowercase band name. A custom
# band name outside this set is shown as-is (capitalized), in whichever
# language, since it can't be translated automatically.
_BAND_DISPLAY_NAMES: Dict[str, Dict[str, str]] = {
    "delta": {"en": "Delta", "pt-br": "Delta"},
    "theta": {"en": "Theta", "pt-br": "Teta"},
    "alpha": {"en": "Alpha", "pt-br": "Alfa"},
    "beta":  {"en": "Beta",  "pt-br": "Beta"},
    "gamma": {"en": "Gamma", "pt-br": "Gama"},
}


def _band_display_name(band_key: str, language: str) -> str:
    info = _BAND_DISPLAY_NAMES.get(band_key.lower())
    return info[language] if info else band_key.capitalize()


def _epoch_count_label(n: int, language: str) -> str:
    if language == "pt-br":
        return f"{n} época" if n == 1 else f"{n} épocas"
    return f"{n} epoch" if n == 1 else f"{n} epochs"


# Plain-language explanations of what a band's power is commonly associated
# with in the literature, keyed by lowercase band name, plus the sources
# that support the claim (checked against 2024-2025 reviews, not only the
# classic papers -- some classic claims, e.g. alpha's inhibition mechanism,
# are actively debated in more recent work and are hedged accordingly).
# Matched by name against whatever keys the caller passes in *bands*; a
# custom/unrecognized name falls back to a neutral, no-claims description.
# References are academic titles and are not translated.
_BAND_KNOWLEDGE: Dict[str, Dict[str, object]] = {
    "delta": {
        "text": {
            "en": (
                "Delta reflects very slow cortical activity. In an awake person "
                "completing a task, higher delta power is most often linked to "
                "drowsiness or reduced attentional engagement, though some studies "
                "also associate frontal delta with inhibiting task-irrelevant "
                "activity. It is also the band most easily confounded with slow "
                "artifacts such as electrode drift or head movement, so a high "
                "delta bar should be read cautiously."
            ),
            "pt-br": (
                "O delta reflete atividade cortical muito lenta. Em uma pessoa "
                "acordada realizando uma tarefa, maior potência delta costuma "
                "estar associada a sonolência ou engajamento atencional "
                "reduzido, embora alguns estudos também associem delta frontal "
                "à inibição de atividade irrelevante à tarefa. É "
                "também a banda mais facilmente confundida com artefatos "
                "lentos, como deriva de eletrodo ou movimento da cabeça, "
                "então uma barra de delta alta deve ser lida com cautela."
            ),
        },
        "refs": [
            ("The functional significance of delta oscillations in cognitive processing (2013)",
             "https://www.frontiersin.org/journals/integrative-neuroscience/articles/10.3389/fnint.2013.00083/full"),
        ],
    },
    "theta": {
        "text": {
            "en": (
                "Theta, especially over frontal-midline sites, is one of the most "
                "consistently replicated EEG markers of active cognitive effort: it "
                "tends to rise with working-memory load, executive control demands, "
                "and the encoding of new information. This association is well "
                "supported, including by studies that manipulate theta directly "
                "(neurofeedback, brain stimulation) rather than only observing it. "
                "A caveat worth keeping in mind: theta also rises during drowsiness "
                "and low arousal, so a high theta bar can reflect either engaged "
                "effort or disengagement -- the two are not distinguishable from "
                "band power alone."
            ),
            "pt-br": (
                "O theta, especialmente em sítios frontais-mediais, é um "
                "dos marcadores de EEG mais consistentemente replicados de esforço "
                "cognitivo ativo: tende a aumentar com a carga de memória de "
                "trabalho, demandas de controle executivo e a codificação de "
                "novas informações. Essa associação é bem "
                "sustentada, inclusive por estudos que manipulam o theta diretamente "
                "(neurofeedback, estimulação cerebral) em vez de apenas "
                "observá-lo. Uma ressalva importante: o theta também aumenta "
                "durante sonolência e baixa vigília, então uma barra de "
                "theta alta pode refletir tanto esforço engajado quanto "
                "desengajamento — os dois não são distinguíveis "
                "apenas pela potência da banda."
            ),
        },
        "refs": [
            ("Modulation of human frontal midline theta by neurofeedback: a systematic review and meta-analysis (2024)",
             "https://pubmed.ncbi.nlm.nih.gov/38723734/"),
            ("Working memory readout varies with frontal theta rhythms (2025)",
             "https://www.cell.com/neuron/abstract/S0896-6273(25)00744-5"),
        ],
    },
    "alpha": {
        "text": {
            "en": (
                "Alpha is classically the rhythm of relaxed wakefulness, strongest "
                "at rest and typically decreasing -- a pattern called event-related "
                "desynchronization -- during active visual or cognitive processing. "
                "A widely cited hypothesis (\"gating by inhibition\") proposes that "
                "alpha actively suppresses processing in task-irrelevant brain "
                "regions, so lower alpha in task-relevant areas would reflect active "
                "engagement. This specific mechanism is actively debated in more "
                "recent work, which finds only limited support for a suppression "
                "account and proposes alternative explanations -- so treat the "
                "engagement pattern itself (alpha tends to drop during active "
                "processing) as more established than the inhibition explanation "
                "for why it happens."
            ),
            "pt-br": (
                "O alfa é classicamente o ritmo do repouso relaxado, mais forte "
                "em repouso e tipicamente diminuindo — um padrão chamado "
                "dessincronização relacionada a evento — durante "
                "processamento visual ou cognitivo ativo. Uma hipótese amplamente "
                "citada (“gating by inhibition”, controle por inibição) "
                "propõe que o alfa suprime ativamente o processamento em regiões "
                "cerebrais irrelevantes à tarefa, de modo que um alfa mais baixo "
                "em áreas relevantes à tarefa refletiria engajamento ativo. "
                "Esse mecanismo específico é ativamente debatido em trabalhos "
                "mais recentes, que encontram suporte apenas limitado para uma "
                "explicação de supressão e propõem explicações "
                "alternativas — portanto, trate o padrão de engajamento em si "
                "(o alfa tende a cair durante o processamento ativo) como mais "
                "estabelecido do que a explicação de inibição para o "
                "motivo de isso acontecer."
            ),
        },
        "refs": [
            ("Shaping Functional Architecture by Oscillatory Alpha Activity: Gating by Inhibition (2010)",
             "https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2010.00186/full"),
            ("The role of alpha oscillations in spatial attention: limited evidence for a suppression account (2018)",
             "https://pmc.ncbi.nlm.nih.gov/articles/PMC6506396/"),
            ("Distractor inhibition by alpha oscillations is controlled by an indirect mechanism (2024)",
             "https://www.nature.com/articles/s44271-024-00081-w"),
        ],
    },
    "beta": {
        "text": {
            "en": (
                "Beta is linked to active concentration and to sensorimotor "
                "processing -- planning and executing responses, which is relevant "
                "here since these tasks involve gaze selection and response "
                "fixation. Like alpha, it tends to decrease during active "
                "engagement; more recent work also frames elevated beta as related "
                "to maintaining the current cognitive or motor state (\"status "
                "quo\") rather than driving a change in behavior. Practically "
                "important: beta power is also one of the bands most easily "
                "inflated by muscle activity (jaw, scalp, or neck tension), which "
                "tends to increase specifically during effortful tasks -- worth "
                "keeping in mind for head-mounted VR setups where the headset "
                "itself adds physical pressure near the electrodes."
            ),
            "pt-br": (
                "O beta está ligado à concentração ativa e ao "
                "processamento sensório-motor — planejamento e execução "
                "de respostas, o que é relevante aqui já que essas tarefas "
                "envolvem seleção de olhar e fixação de resposta. "
                "Assim como o alfa, tende a diminuir durante o engajamento ativo; "
                "trabalhos mais recentes também associam o beta elevado à "
                "manutenção do estado cognitivo ou motor atual (“status "
                "quo”) em vez de impulsionar uma mudança de comportamento. "
                "Importante na prática: a potência do beta também é "
                "uma das mais facilmente infladas por atividade muscular (tensão "
                "de mandíbula, couro cabeludo ou pescoço), que tende a "
                "aumentar especificamente durante tarefas efortosas — vale "
                "considerar em configurações de VR com headset, onde o "
                "próprio equipamento adiciona pressão física perto dos "
                "eletrodos."
            ),
        },
        "refs": [
            ("Beta-band oscillations — signalling the status quo? (2010)",
             "https://pubmed.ncbi.nlm.nih.gov/20359884/"),
            ("Beta: bursts of cognition (2024)",
             "https://www.sciencedirect.com/science/article/pii/S1364661324000779"),
            ("High-frequency brain activity and muscle artifacts in MEG/EEG: a review and recommendations (2013)",
             "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3625857/"),
        ],
    },
    "gamma": {
        "text": {
            "en": (
                "Gamma is associated in the literature with binding features into a "
                "coherent percept and with higher-order integrative and "
                "working-memory processes. In practice, scalp-recorded gamma is "
                "heavily confounded by muscle activity: facial, jaw, and scalp "
                "muscles produce electrical signals that overlap the same "
                "frequency range and are often far stronger than the underlying "
                "brain signal, and this contamination increases specifically "
                "during effortful cognitive tasks. Without dedicated "
                "muscle-artifact rejection, a high gamma bar more often reflects "
                "muscle tension than cortical activity, which is why this band is "
                "not part of the report's default set."
            ),
            "pt-br": (
                "O gama é associado na literatura à ligação de "
                "características em uma percepção coerente e a "
                "processos integrativos de ordem superior e de memória de "
                "trabalho. Na prática, o gama registrado no couro cabeludo é "
                "fortemente confundido por atividade muscular: músculos "
                "faciais, da mandíbula e do couro cabeludo produzem sinais "
                "elétricos que se sobrepõem à mesma faixa de "
                "frequência e costumam ser muito mais fortes que o sinal "
                "cerebral subjacente, e essa contaminação aumenta "
                "especificamente durante tarefas cognitivas efortosas. Sem "
                "rejeição dedicada de artefato muscular, uma barra de gama "
                "alta reflete mais frequentemente tensão muscular do que "
                "atividade cortical, motivo pelo qual essa banda não faz parte "
                "do conjunto padrão do relatório."
            ),
        },
        "refs": [
            ("High-frequency brain activity and muscle artifacts in MEG/EEG: a review and recommendations (2013)",
             "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3625857/"),
            ("Managing electromyogram contamination in scalp recordings (2021)",
             "https://www.medrxiv.org/content/10.1101/2021.11.18.21265963.full.pdf"),
        ],
    },
}

# Categorical colors (validated for CVD-safe adjacent contrast; see the
# project's data-viz palette). Slot 1 is reserved for the cognitive score;
# bands are assigned slots 2+ in the order they appear in *bands*.
_SCORE_COLOR = "#2a78d6"
_BAND_COLORS = [
    "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
    "#008300", "#4a3aa7", "#e34948",
]

# Region prefixes from the standard 10-20/10-10 EEG electrode nomenclature,
# ordered longest-code-first so e.g. "FC" matches before the shorter "F".
# Used to label electrodes by scalp region regardless of which montage a
# given recording uses -- nothing here is specific to any one headset.
# The English name on the right is the internal/canonical key used
# elsewhere (e.g. by _classify_electrode); display strings are looked up
# per language via _REGION_DISPLAY_NAMES.
_REGION_PREFIXES = [
    ("FP", "Frontopolar"),
    ("AF", "Anterior Frontal"),
    ("FC", "Fronto-Central"),
    ("FT", "Fronto-Temporal"),
    ("CP", "Centro-Parietal"),
    ("TP", "Temporo-Parietal"),
    ("PO", "Parieto-Occipital"),
    ("F",  "Frontal"),
    ("C",  "Central"),
    ("T",  "Temporal"),
    ("P",  "Parietal"),
    ("O",  "Occipital"),
    ("M",  "Mastoid"),
    ("A",  "Mastoid/Ear reference"),
]

# Anteroposterior reading order for the legend -- distinct from
# _REGION_PREFIXES above, whose order instead encodes match priority
# (longest code first, so "FC" is tried before "F").
_REGION_DISPLAY_ORDER = [
    "Frontopolar", "Anterior Frontal", "Frontal", "Fronto-Central",
    "Fronto-Temporal", "Central", "Centro-Parietal", "Temporal",
    "Temporo-Parietal", "Parietal", "Parieto-Occipital", "Occipital",
    "Mastoid", "Mastoid/Ear reference",
]

_REGION_DISPLAY_NAMES: Dict[str, Dict[str, str]] = {
    "Frontopolar":            {"en": "Frontopolar",            "pt-br": "Frontopolar"},
    "Anterior Frontal":       {"en": "Anterior Frontal",       "pt-br": "Frontal Anterior"},
    "Frontal":                {"en": "Frontal",                "pt-br": "Frontal"},
    "Fronto-Central":         {"en": "Fronto-Central",         "pt-br": "Fronto-Central"},
    "Fronto-Temporal":        {"en": "Fronto-Temporal",        "pt-br": "Fronto-Temporal"},
    "Central":                {"en": "Central",                "pt-br": "Central"},
    "Centro-Parietal":        {"en": "Centro-Parietal",        "pt-br": "Centro-Parietal"},
    "Temporal":                {"en": "Temporal",               "pt-br": "Temporal"},
    "Temporo-Parietal":       {"en": "Temporo-Parietal",       "pt-br": "Temporo-Parietal"},
    "Parietal":                {"en": "Parietal",               "pt-br": "Parietal"},
    "Parieto-Occipital":      {"en": "Parieto-Occipital",      "pt-br": "Parieto-Occipital"},
    "Occipital":               {"en": "Occipital",              "pt-br": "Occipital"},
    "Mastoid":                 {"en": "Mastoid",                "pt-br": "Mastóideo"},
    "Mastoid/Ear reference":   {"en": "Mastoid/Ear reference",  "pt-br": "Mastóideo/Referência auricular"},
}

# All user-facing strings, keyed by a symbolic name and then by language.
# `{placeholder}` tokens are filled in via str.format() at render time.
_STRINGS: Dict[str, Dict[str, str]] = {
    "title": {
        "en": "EVA Results Summary &mdash; {stem}",
        "pt-br": "Resumo de Resultados da EVA &mdash; {stem}",
    },
    "h1": {
        "en": "Results Summary",
        "pt-br": "Resumo de Resultados",
    },
    "meta": {
        "en": "Generated by EVA &mdash; EEG data eValuation and preprocessing Assistant &middot; recording: {stem}",
        "pt-br": "Gerado pela EVA &mdash; EEG data eValuation and preprocessing Assistant &middot; gravação: {stem}",
    },
    "note_intro": {
        "en": (
            'This report compares each task/condition using one bar per measurement: '
            'how well the participant scored, and how much brain activity was measured '
            'in each of the following frequency ranges during that task: {band_list}. '
            '<strong>Every bar shows a 0&ndash;100 scale relative to this recording '
            'session only</strong> &mdash; none are compared to any clinical reference, '
            'age norm, or population average. A short bar simply means "lower than '
            'other tasks in this same session", not "abnormal".'
        ),
        "pt-br": (
            'Este relatório compara cada tarefa/condição usando uma barra '
            'por medida: o quão bem o participante pontuou, e o quanto de atividade '
            'cerebral foi medida em cada uma das seguintes faixas de frequência '
            'durante aquela tarefa: {band_list}. <strong>Cada barra mostra uma escala '
            'de 0&ndash;100 relativa apenas a esta sessão de gravação</strong> '
            '&mdash; nenhuma é comparada a referência clínica, norma '
            'etária ou média populacional. Uma barra curta significa apenas '
            '"mais baixa que outras tarefas nesta mesma sessão", não "anormal".'
        ),
    },
    "warn_note": {
        "en": (
            'A taller activity bar is <strong>not</strong> simply "better" or "worse", '
            'and the meaning is not the same across bands: in the research literature, '
            'higher power in slower bands (e.g. theta) is often associated with '
            'effortful engagement, while <em>lower</em> power in faster bands (e.g. '
            'alpha/beta &mdash; a pattern called desynchronization) is often associated '
            'with active engagement instead. This report does not test or confirm '
            'either interpretation for this recording &mdash; it is descriptive only, '
            'not a diagnostic tool, and does not test whether differences between '
            'tasks are statistically meaningful. Total epochs analysed: '
            '{n_epochs_total} across {n_conditions} condition(s).'
        ),
        "pt-br": (
            'Uma barra de atividade mais alta <strong>não</strong> significa '
            'simplesmente "melhor" ou "pior", e o significado não é o mesmo '
            'entre bandas: na literatura de pesquisa, maior potência em bandas '
            'mais lentas (ex. theta) costuma estar associada a engajamento efortoso, '
            'enquanto potência <em>mais baixa</em> em bandas mais rápidas '
            '(ex. alfa/beta &mdash; um padrão chamado dessincronização) '
            'costuma estar associada a engajamento ativo. Este relatório não '
            'testa nem confirma nenhuma das duas interpretações para esta '
            'gravação &mdash; é apenas descritivo, não é uma '
            'ferramenta diagnóstica, e não testa se as diferenças entre '
            'tarefas são estatisticamente significativas. Total de épocas '
            'analisadas: {n_epochs_total} em {n_conditions} condição(ões).'
        ),
    },
    "how_calculated": {
        "en": (
            '<strong>How are these numbers calculated?</strong> For each task, we take '
            'the EEG signal recorded from all electrodes and measure how strong the '
            'electrical activity was within the chosen frequency range (theta, alpha, '
            'or beta) &mdash; averaged across all electrodes, and across repeated '
            'attempts of the same task when there was more than one. That gives one '
            '"brain activity" number per task, and the <em>Cognitive score</em> bar is '
            "built the same way, using the participant's task score instead of EEG "
            'power. Each of those numbers is then compared only against the other '
            'tasks in this same recording: the task with the highest value becomes '
            '100, the lowest becomes 0, and the rest fall in between. These numbers '
            'cannot be compared across different people or different recordings, only '
            'within the tasks shown here.'
        ),
        "pt-br": (
            '<strong>Como esses números são calculados?</strong> Para cada '
            'tarefa, pegamos o sinal de EEG registrado em todos os eletrodos e medimos '
            'o quão forte foi a atividade elétrica dentro da faixa de '
            'frequência escolhida (theta, alfa ou beta) &mdash; em média entre '
            'todos os eletrodos, e entre tentativas repetidas da mesma tarefa quando '
            'houve mais de uma. Isso gera um número de "atividade cerebral" por '
            'tarefa, e a barra de <em>Pontuação cognitiva</em> é '
            'construída da mesma forma, usando o escore da tarefa do participante '
            'em vez da potência de EEG. Cada um desses números é então '
            'comparado apenas com as outras tarefas desta mesma gravação: a '
            'tarefa com o valor mais alto vira 100, a mais baixa vira 0, e as demais '
            'ficam entre os dois. Esses números não podem ser comparados '
            'entre pessoas diferentes ou gravações diferentes, apenas entre '
            'as tarefas mostradas aqui.'
        ),
    },
    "h2_by_task": {"en": "By Task / Condition", "pt-br": "Por Tarefa / Condição"},
    "card_score_label": {"en": "Cognitive score", "pt-br": "Pontuação cognitiva"},
    "card_activity_label": {"en": "{band} activity", "pt-br": "Atividade {band}"},
    "h2_glossary": {
        "en": "What Do These Bands Mean?",
        "pt-br": "O Que Essas Bandas Significam?",
    },
    "glossary_intro": {
        "en": (
            'These summaries describe what each frequency band is commonly associated '
            'with in the research literature in general &mdash; they are background to '
            'help you read the bars above, not a claim about this specific recording. '
            'Sources are linked so you can check them yourself.'
        ),
        "pt-br": (
            'Esses resumos descrevem com o que cada banda de frequência costuma '
            'ser associada na literatura de pesquisa em geral &mdash; são contexto '
            'para ajudar a interpretar as barras acima, não uma afirmação '
            'sobre esta gravação específica. As fontes estão '
            'linkadas para você conferir por conta própria.'
        ),
    },
    "technical_summary": {"en": "Technical details", "pt-br": "Detalhes técnicos"},
    "technical_note": {
        "en": (
            'Raw (non-normalised) values as stored in the file: behavioral score as '
            'provided to <code>sync()</code>, and band power in V&sup2;/Hz averaged '
            'across channels and epochs.'
        ),
        "pt-br": (
            'Valores brutos (não normalizados) como armazenados no arquivo: '
            'escore comportamental fornecido a <code>sync()</code>, e potência '
            'de banda em V&sup2;/Hz, em média entre canais e épocas.'
        ),
    },
    "th_condition": {"en": "Condition", "pt-br": "Condição"},
    "th_raw_label": {"en": "Raw label", "pt-br": "Rótulo bruto"},
    "th_epochs": {"en": "Epochs", "pt-br": "Épocas"},
    "th_score_raw": {"en": "Score (raw)", "pt-br": "Escore (bruto)"},
    "th_band_power_raw": {"en": "{band} power (raw)", "pt-br": "Potência {band} (bruta)"},
    "electrode_note": {
        "en": (
            'Per-electrode power (&micro;V&sup2;/Hz), for readers who want scalp '
            'location instead of the whole-head average shown above. Region labels '
            'are inferred from standard 10-20/10-10 electrode names; electrodes that '
            "don't follow this naming convention are listed without a region. "
            'Electrode groups in this recording: {electrode_legend}.'
        ),
        "pt-br": (
            'Potência por eletrodo (&micro;V&sup2;/Hz), para quem preferir a '
            'localização no couro cabeludo em vez da média de toda a '
            'cabeça mostrada acima. Os rótulos de região são '
            'inferidos a partir dos nomes de eletrodo padrão 10-20/10-10; '
            'eletrodos que não seguem essa convenção são listados '
            'sem região. Grupos de eletrodos nesta gravação: '
            '{electrode_legend}.'
        ),
    },
    "electrode_table_title": {
        "en": "{band} power by electrode",
        "pt-br": "Potência {band} por eletrodo",
    },
    "not_classified": {"en": "Not classified", "pt-br": "Não classificado"},
    "footer": {
        "en": "EVA &mdash; EEG data eValuation and preprocessing Assistant",
        "pt-br": "EVA &mdash; EEG data eValuation and preprocessing Assistant",
    },
    "custom_band_fallback": {
        "en": (
            'No established functional interpretation is bundled for this custom '
            'band. Treat it as a purely descriptive spectral summary rather than a '
            'marker with a known cognitive meaning.'
        ),
        "pt-br": (
            'Não há interpretação funcional estabelecida associada '
            'a essa banda personalizada. Trate-a como um resumo espectral puramente '
            'descritivo, não como um marcador com significado cognitivo '
            'conhecido.'
        ),
    },
}


def _s(key: str, language: str, **kwargs) -> str:
    text = _STRINGS[key][language]
    return text.format(**kwargs) if kwargs else text


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def interpret(
    path: Union[str, Path],
    *,
    score_key: str = "score",
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    label_names: Optional[Dict[str, str]] = None,
    language: str = "en",
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Generate a plain-language HTML report relating behavioral scores to
    EEG activity, grouped by condition.

    Unlike :func:`eva.preprocess`'s technical quality report, this report
    is written for readers without a signal-processing background: no
    filter parameters, spectral entropy, or inferential statistics are
    shown. Every value is expressed as a 0-100 relative level, computed
    within the session itself.

    Parameters
    ----------
    path
        Path to a .h5 file already processed by :func:`eva.preprocess`
        and synchronised with behavioral scores via :func:`eva.sync`.
    score_key
        Key under ``/behavioral/`` holding the numeric score per epoch
        (e.g. ``"score"``, ``"accuracy"``).
    bands
        Mapping of band name to ``(fmin, fmax)`` in Hz, summarised as
        "brain activity level" per band. Defaults to
        ``{"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}``.
    label_names
        Optional mapping from raw condition labels (as stored in
        ``/eeg/label_names``) to human-readable names, e.g.
        ``{"vr_att": "Attention"}``. Conditions absent from the mapping
        are shown with their raw label.
    language
        Report language: ``"en"`` (default) or ``"pt-br"``, mirroring
        VECA-EEG's own English/Portuguese support. Only the report's own
        text is translated; task/condition names passed via *label_names*
        and academic references are shown as given.
    output_dir
        Destination directory for the report. Defaults to a ``results``
        folder next to *path*.

    Returns
    -------
    Path
        Location of the saved HTML file: ``<stem>_results.html`` for
        English, ``<stem>_results_<language>.html`` otherwise.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If *language* is not supported, if the file lacks an ``/eeg`` or
        ``/behavioral`` group, if *score_key* is not present under
        ``/behavioral``, or if any band is invalid for the recording's
        sampling rate.
    """
    import h5py

    if language not in _SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{language}'. "
            f"Supported: {', '.join(_SUPPORTED_LANGUAGES)}."
        )

    bands = bands or _DEFAULT_BANDS
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with h5py.File(path, "r") as f:
        if "eeg" not in f:
            raise ValueError(f"'{path.name}' has no '/eeg/' group.")
        if "behavioral" not in f or score_key not in f["behavioral"]:
            raise ValueError(
                f"'{path.name}' has no '/behavioral/{score_key}' dataset. "
                "Run sync() with this score before calling interpret()."
            )

        sfreq = float(f["metadata"].attrs["sfreq"])
        _validate_bands(bands, sfreq)

        eeg_data         = f["eeg/data"][:]
        labels           = f["eeg/labels"][:]
        label_names_raw  = _decode_strings(f["eeg/label_names"][:])
        label_codes      = f["eeg/label_codes"][:]
        ch_names         = _decode_strings(f["eeg/ch_names"][:])
        scores           = f[f"behavioral/{score_key}"][:]

    code_to_name = dict(zip(label_codes.tolist(), label_names_raw))
    conditions = np.array([code_to_name[int(c)] for c in labels])

    band_activity_by_channel = _band_power_per_epoch_by_channel(eeg_data, sfreq, bands)
    band_activity = {name: arr.mean(axis=1) for name, arr in band_activity_by_channel.items()}
    summary = _summarize(conditions, scores, band_activity, label_names)
    per_channel_summary = _summarize_per_channel(
        conditions, ch_names, band_activity_by_channel, label_names
    )

    stem = path.stem
    out_dir = Path(output_dir) if output_dir else path.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(out_dir / f"{stem}_summary.csv", index=False)

    html = _build_html(stem, summary, bands, len(conditions), ch_names, per_channel_summary, language)
    suffix = "" if language == "en" else f"_{language}"
    html_path = out_dir / f"{stem}_results{suffix}.html"
    html_path.write_text(html, encoding="utf-8")

    logger.info("Results report saved -> %s", html_path)
    return html_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_bands(bands: Dict[str, Tuple[float, float]], sfreq: float) -> None:
    nyquist = sfreq / 2.0
    for name, (fmin, fmax) in bands.items():
        if fmin >= fmax:
            raise ValueError(
                f"Invalid frequency band '{name}' {(fmin, fmax)}: the lower "
                "bound must be less than the upper bound."
            )
        if fmax > nyquist:
            raise ValueError(
                f"Band '{name}' {(fmin, fmax)} exceeds the Nyquist frequency "
                f"({nyquist:g} Hz) for this recording (sfreq={sfreq:g} Hz)."
            )


def _decode_strings(arr: np.ndarray) -> list:
    """h5py variable-length string datasets may come back as bytes or str."""
    return [s.decode() if isinstance(s, bytes) else s for s in arr]


def _band_power_per_epoch_by_channel(
    data: np.ndarray,
    sfreq: float,
    bands: Dict[str, Tuple[float, float]],
) -> Dict[str, np.ndarray]:
    """
    Mean power spectral density within each band, per channel, for each
    epoch. The PSD is computed once per epoch and reused across all bands.

    Parameters
    ----------
    data  : (n_epochs, n_channels, n_samples)
    sfreq : sampling frequency (Hz)
    bands : mapping of band name to (fmin, fmax) in Hz

    Returns
    -------
    dict mapping band name to an (n_epochs, n_channels) array in V^2/Hz.
    """
    n_epochs, n_channels, n_samples = data.shape
    nperseg = min(256, n_samples)

    power = {name: np.empty((n_epochs, n_channels)) for name in bands}
    for i in range(n_epochs):
        freqs, psd = compute_psd(data[i], sfreq, nperseg=nperseg)
        for name, (fmin, fmax) in bands.items():
            mask = (freqs >= fmin) & (freqs <= fmax)
            power[name][i] = psd[:, mask].mean(axis=1) if mask.any() else np.full(n_channels, np.nan)
    return power


def _band_power_per_epoch(
    data: np.ndarray,
    sfreq: float,
    bands: Dict[str, Tuple[float, float]],
) -> Dict[str, np.ndarray]:
    """
    Mean power spectral density within each band, averaged across channels,
    for each epoch. Thin wrapper over :func:`_band_power_per_epoch_by_channel`.

    Returns
    -------
    dict mapping band name to an (n_epochs,) array in V^2/Hz.
    """
    by_channel = _band_power_per_epoch_by_channel(data, sfreq, bands)
    return {name: arr.mean(axis=1) for name, arr in by_channel.items()}


def _classify_electrode(ch_name: str) -> Optional[str]:
    """
    Infer the coarse scalp region from a standard 10-20/10-10 electrode
    name, e.g. "FC1" -> "Fronto-Central", "Cz" -> "Central", "P4" -> "Parietal".

    Returns ``None`` for names that don't follow the convention (proprietary
    or numeric channel labels, reference/ground channels such as "REF" or
    "GND") so the report never guesses at a region it can't actually infer.
    This makes the report montage-agnostic: it adapts to whatever channels
    are in the file rather than assuming any particular electrode layout.
    """
    prefix = ch_name.strip().rstrip("0123456789").rstrip("zZ").upper()
    for code, region in _REGION_PREFIXES:
        if prefix == code:
            return region
    return None


def _minmax_0_100(values: np.ndarray) -> np.ndarray:
    """Rescale to [0, 100]; returns 50 for every value when input is constant."""
    values = np.asarray(values, dtype=float)
    lo, hi = np.nanmin(values), np.nanmax(values)
    if hi - lo < 1e-30:
        return np.full_like(values, 50.0)
    return (values - lo) / (hi - lo) * 100.0


def _summarize(
    conditions: np.ndarray,
    scores: np.ndarray,
    band_activity: Dict[str, np.ndarray],
    label_names: Optional[Dict[str, str]],
) -> pd.DataFrame:
    """
    One row per *display* condition, with raw and 0-100 relative-level
    columns for the score and for each band (``"{band}_raw"``,
    ``"{band}_level"``).

    Grouping happens on the mapped display name, not the raw label, so that
    several raw labels sharing one display name (e.g. three memory-task
    features all mapped to "Memory") are consolidated into a single row
    instead of appearing as duplicate cards.
    """
    mapping = label_names or {}
    display = np.array([mapping.get(c, c) for c in conditions])

    df = pd.DataFrame({
        "condition_raw": conditions,
        "condition":     display,
        "score_raw":     np.asarray(scores, dtype=float),
    })
    df["score_level"] = _minmax_0_100(df["score_raw"].values)

    agg = {
        "n_epochs": ("score_raw", "size"),
        "score_raw": ("score_raw", "mean"),
        "score_level": ("score_level", "mean"),
        "condition_raw": ("condition_raw", lambda s: ", ".join(sorted(set(s)))),
    }
    for band_name, raw_values in band_activity.items():
        df[f"{band_name}_raw"] = raw_values
        df[f"{band_name}_level"] = _minmax_0_100(raw_values)
        agg[f"{band_name}_raw"] = (f"{band_name}_raw", "mean")
        agg[f"{band_name}_level"] = (f"{band_name}_level", "mean")

    grouped = df.groupby("condition", sort=False).agg(**agg).reset_index()

    ordered_cols = ["condition", "condition_raw", "n_epochs", "score_raw", "score_level"]
    for band_name in band_activity:
        ordered_cols += [f"{band_name}_raw", f"{band_name}_level"]
    return grouped[ordered_cols]


def _summarize_per_channel(
    conditions: np.ndarray,
    ch_names: List[str],
    band_activity_by_channel: Dict[str, np.ndarray],
    label_names: Optional[Dict[str, str]],
) -> Dict[str, pd.DataFrame]:
    """
    Per-band, per-electrode raw power (V^2/Hz) averaged within each display
    condition. Mirrors the grouping logic in :func:`_summarize` (raw labels
    sharing a display name are consolidated, same row order), returned as
    one DataFrame per band with conditions as the index and channels as
    columns.
    """
    mapping = label_names or {}
    display = np.array([mapping.get(c, c) for c in conditions])

    result = {}
    for band_name, arr in band_activity_by_channel.items():
        df = pd.DataFrame(arr, columns=ch_names)
        df["condition"] = display
        result[band_name] = df.groupby("condition", sort=False).mean()
    return result


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="{lang_attr}">
<head>
<meta charset="UTF-8"/>
<title>{title}</title>
<style>
  body    {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px 60px;
             background: #f4f6f9; color: #212529; line-height: 1.6; }}
  h1      {{ color: #1a252f; margin-bottom: 4px; }}
  h2      {{ color: #2c3e50; border-bottom: 2px solid #bdc3c7;
             padding-bottom: 5px; margin-top: 36px; }}
  h4      {{ color: #2c3e50; margin: 22px 0 4px; font-size: 13px; }}
  .meta   {{ color: #5d6d7e; font-size: 14px; margin-bottom: 6px; }}
  .note   {{ background: #eaf4fb; border-left: 4px solid #3498db;
             padding: 10px 16px; margin: 12px 0; border-radius: 3px;
             font-size: 13px; color: #1a5276; }}
  .warn-note {{ background: #fef9e7; border-left: 4px solid #f39c12;
             padding: 10px 16px; margin: 12px 0; border-radius: 3px;
             font-size: 13px; color: #7d6608; }}
  .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 14px; margin: 16px 0; }}
  .card   {{ background: #fff; border-radius: 8px; padding: 16px 20px;
             border: 1px solid #dee2e6; box-shadow: 0 1px 3px rgba(0,0,0,0.07); }}
  .card-title {{ font-size: 16px; font-weight: 700; color: #2c3e50; margin-bottom: 10px; }}
  .card-meta  {{ font-size: 11px; color: #95a5a6; margin-top: 6px; }}
  .bar-row    {{ display: flex; align-items: center; gap: 8px; margin: 6px 0; }}
  .bar-label  {{ font-size: 12px; color: #5d6d7e; width: 130px; flex-shrink: 0; }}
  .bar-track  {{ flex: 1; background: #ecf0f1; border-radius: 4px; height: 14px; overflow: hidden; }}
  .bar-fill   {{ height: 100%; border-radius: 4px; }}
  .bar-value  {{ font-size: 12px; color: #2c3e50; width: 48px; text-align: right; flex-shrink: 0; }}
  table   {{ border-collapse: collapse; width: 100%; margin-bottom: 18px;
             font-size: 13px; background: #fff; }}
  th, td  {{ border: 1px solid #dee2e6; padding: 6px 10px; text-align: right; }}
  th      {{ background: #2c3e50; color: #ecf0f1; text-align: left; font-weight: 600; }}
  td:first-child, td:nth-child(2) {{ text-align: left; }}
  tr:nth-child(even) {{ background: #f2f3f4; }}
  details {{ margin-top: 30px; }}
  summary {{ cursor: pointer; font-weight: 600; color: #2c3e50; padding: 8px 0; }}
  .glossary h3 {{ color: #2c3e50; margin: 18px 0 4px; font-size: 14px; }}
  .glossary p  {{ margin: 4px 0; font-size: 13px; }}
  .glossary ul.refs {{ margin: 4px 0 0; padding-left: 18px; font-size: 12px; }}
  .glossary ul.refs a {{ color: #2a78d6; }}
  footer  {{ margin-top: 50px; color: #aab7b8; font-size: 12px;
             border-top: 1px solid #dee2e6; padding-top: 10px; }}
</style>
</head>
<body>

<h1>{h1}</h1>
<p class="meta">{meta}</p>

<div class="note">
  {note_intro}
</div>
<div class="warn-note">
  {warn_note}
</div>
<div class="note">
  {how_calculated}
</div>

<h2>{h2_by_task}</h2>
<div class="card-grid">
{cards_html}
</div>

<h2>{h2_glossary}</h2>
<div class="note">
  {glossary_intro}
</div>
<div class="glossary">
{glossary_html}
</div>

<details>
<summary>{technical_summary}</summary>
<div class="note">
  {technical_note}
</div>
<table>
<thead><tr>{technical_header}</tr></thead>
<tbody>
{technical_rows}
</tbody>
</table>

<div class="note">
  {electrode_note}
</div>
{electrode_tables}
</details>

<footer>{footer}</footer>
</body>
</html>
"""


def _card_html(row: dict, band_names: List[str], language: str) -> str:
    epoch_label = _epoch_count_label(row["n_epochs"], language)
    bar_rows = (
        f'<div class="bar-row"><span class="bar-label">{_s("card_score_label", language)}</span>'
        f'<div class="bar-track"><div class="bar-fill" '
        f'style="width:{row["score_level"]:.0f}%;background:{_SCORE_COLOR}"></div></div>'
        f'<span class="bar-value">{row["score_level"]:.0f}/100</span></div>'
    )
    for i, band_name in enumerate(band_names):
        color = _BAND_COLORS[i % len(_BAND_COLORS)]
        level = row[f"{band_name}_level"]
        label = _s("card_activity_label", language, band=_band_display_name(band_name, language))
        bar_rows += (
            f'<div class="bar-row"><span class="bar-label">{label}</span>'
            f'<div class="bar-track"><div class="bar-fill" '
            f'style="width:{level:.0f}%;background:{color}"></div></div>'
            f'<span class="bar-value">{level:.0f}/100</span></div>'
        )
    return (
        f'<div class="card">'
        f'<div class="card-title">{row["condition"]}</div>'
        f'{bar_rows}'
        f'<div class="card-meta">{epoch_label}</div>'
        f'</div>'
    )


def _glossary_html(band_names: List[str], language: str) -> str:
    """
    Plain-language explanation of each band actually used in the report,
    with hyperlinked sources so a reader can verify the claim themselves.
    Bands without a bundled explanation (custom ranges) get a neutral
    fallback instead of an invented claim.
    """
    entries = []
    for name in band_names:
        info = _BAND_KNOWLEDGE.get(name.lower())
        display_name = _band_display_name(name, language)
        if info is None:
            entries.append(
                f"<h3>{display_name}</h3>"
                f"<p>{_s('custom_band_fallback', language)}</p>"
            )
            continue
        refs_html = "".join(
            f'<li><a href="{url}" target="_blank" rel="noopener noreferrer">{label}</a></li>'
            for label, url in info["refs"]
        )
        entries.append(
            f"<h3>{display_name}</h3>"
            f"<p>{info['text'][language]}</p>"
            f"<ul class=\"refs\">{refs_html}</ul>"
        )
    return "".join(entries)


def _electrode_legend_html(ch_names: List[str], language: str) -> str:
    """
    Group channel names by inferred region (in a fixed, readable order),
    listing anything that doesn't match the naming convention separately.
    Built fresh from whatever channels are actually in the file, so it
    adapts to any montage instead of assuming a fixed electrode layout.
    """
    by_region: Dict[str, List[str]] = {}
    unclassified: List[str] = []
    for ch in ch_names:
        region = _classify_electrode(ch)
        if region:
            by_region.setdefault(region, []).append(ch)
        else:
            unclassified.append(ch)

    parts = [
        f"{_REGION_DISPLAY_NAMES[region][language]}: {', '.join(by_region[region])}"
        for region in _REGION_DISPLAY_ORDER if region in by_region
    ]
    if unclassified:
        parts.append(f"{_s('not_classified', language)}: {', '.join(unclassified)}")
    return " &middot; ".join(parts)


def _channel_table_html(band_name: str, table: pd.DataFrame, language: str) -> str:
    header = f"<th>{_s('th_condition', language)}</th>" + "".join(
        f"<th>{ch}</th>" for ch in table.columns
    )
    rows = []
    for condition, row in table.iterrows():
        cells = f"<td>{condition}</td>" + "".join(f"<td>{v * 1e12:.3g}</td>" for v in row)
        rows.append(f"<tr>{cells}</tr>")
    title = _s("electrode_table_title", language, band=_band_display_name(band_name, language))
    return (
        f"<h4>{title}</h4>"
        f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _technical_row_html(row: dict, band_names: List[str]) -> str:
    cells = (
        f"<td>{row['condition']}</td><td>{row['condition_raw']}</td>"
        f"<td>{row['n_epochs']}</td><td>{row['score_raw']:.4g}</td>"
    )
    for band_name in band_names:
        cells += f"<td>{row[f'{band_name}_raw']:.4g}</td>"
    return f"<tr>{cells}</tr>"


def _build_html(
    stem: str,
    summary: pd.DataFrame,
    bands: Dict[str, Tuple[float, float]],
    n_epochs_total: int,
    ch_names: List[str],
    per_channel_summary: Dict[str, pd.DataFrame],
    language: str = "en",
) -> str:
    band_names = list(bands.keys())
    records = summary.to_dict("records")

    cards_html = "".join(_card_html(row, band_names, language) for row in records)
    technical_rows = "".join(_technical_row_html(row, band_names) for row in records)

    band_list = "; ".join(
        f"{_band_display_name(name, language)} ({fmin:g}&ndash;{fmax:g}&nbsp;Hz)"
        for name, (fmin, fmax) in bands.items()
    )
    technical_header = (
        f"<th>{_s('th_condition', language)}</th>"
        f"<th>{_s('th_raw_label', language)}</th>"
        f"<th>{_s('th_epochs', language)}</th>"
        f"<th>{_s('th_score_raw', language)}</th>"
    ) + "".join(
        f"<th>{_s('th_band_power_raw', language, band=_band_display_name(name, language))}</th>"
        for name in band_names
    )
    glossary_html = _glossary_html(band_names, language)
    electrode_legend = _electrode_legend_html(ch_names, language)
    electrode_tables = "".join(
        _channel_table_html(name, per_channel_summary[name], language) for name in band_names
    )

    return _HTML_TEMPLATE.format(
        lang_attr=_HTML_LANG_ATTR[language],
        title=_s("title", language, stem=stem),
        h1=_s("h1", language),
        meta=_s("meta", language, stem=stem),
        note_intro=_s("note_intro", language, band_list=band_list),
        warn_note=_s("warn_note", language, n_epochs_total=n_epochs_total, n_conditions=len(summary)),
        how_calculated=_s("how_calculated", language),
        h2_by_task=_s("h2_by_task", language),
        cards_html=cards_html,
        h2_glossary=_s("h2_glossary", language),
        glossary_intro=_s("glossary_intro", language),
        glossary_html=glossary_html,
        technical_summary=_s("technical_summary", language),
        technical_note=_s("technical_note", language),
        technical_header=technical_header,
        technical_rows=technical_rows,
        electrode_note=_s("electrode_note", language, electrode_legend=electrode_legend),
        electrode_tables=electrode_tables,
        footer=_s("footer", language),
    )
