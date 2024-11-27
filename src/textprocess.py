import nltk
from nltk.tokenize import wordpunct_tokenize
import re
import string


def remove_url(text: str) -> list:
    """Remove a ocorrencia de urls em blocos de texto.

    Args:
    text(str): bloco de texto que você quer remover os urls.

    Returns:
        list: lista com cada letra.
    """

    output = []

    http_str = (
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|"
        + r"[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )

    www_str = (
        r"www?.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|"
        + r"(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )

    http_exp = re.compile(http_str)
    www_exp = re.compile(www_str)

    url_exps = [http_exp, www_exp]

    for line in text:
        for exp in url_exps:
            urls = exp.findall(line)
            for u in urls:
                line = line.replace(u, " ")
        output.append(line)

    return output


def remove_regex(text, regex_pattern):
    """
    remove um dado padrão regex
    """

    output = []

    for line in text:
        matches = re.finditer(regex_pattern, line)

        for m in matches:
            line = re.sub(m.group().strip(), "", line)

        output.append(line)

    return output


def remove_emoticons(text: str) -> list:
    """Remove emoticons de um dado texto

    Args:
    text(str): Texto no qual quer remover emoticons

    Returns:
        list: lista de caracteres sem emoticon
    """

    emoticon_regex = re.compile(
        pattern="["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+",
        flags=re.UNICODE,
    )

    output = []

    for line in text:
        line = emoticon_regex.sub(r"", line)

        output.append(line)

    return output


def remove_punctuation(text):
    """Remove emoticons de um dado texto

    Args:
        list: lista de textos nos quais quer remover pontuacao

    Returns:
        list: lista de textos sem pontuacao

    """

    output = []

    for line in text:
        line = line.translate(str.maketrans("", "", string.punctuation + "’"))
        output.append(line)

    return output


def tokenize_text(text):
    """
    tokeniza
    """

    output = []

    for line in text:
        tokens = wordpunct_tokenize(line)
        output.append(tokens)

    return output


def apply_standardization(text: str, std_list):
    """
    padroniza

    exemplo de std_list : std_list = {'eh': 'é', 'vc': 'você' ...}
    """

    output = []

    for tk_line in text:
        new_tokens = []

        for word in tk_line:
            if word.lower() in std_list:
                word = std_list[word.lower()]

            new_tokens.append(word)

        output.append(new_tokens)

    return output


def remove_stopwords(tokens, stopword_list):
    """
    remove palavras de passagem
    """

    output = []

    for tk_line in tokens:
        new_tokens = []

        for word in tk_line:
            if word.lower() not in stopword_list:
                new_tokens.append(word)

        output.append(new_tokens)

    return output


def apply_stemmer(tokens):
    """
    Aplica o stemmer aos tokes
    """

    output = []
    stemmer = nltk.stem.RSLPStemmer()

    for tk_line in tokens:
        new_tokens = []

        for word in tk_line:
            word = str(stemmer.stem(word))
            new_tokens.append(word)

        output.append(new_tokens)

    return output


def untokenize_text(tokens):
    """
    destokeniza
    """

    output = []

    for tk_line in tokens:
        new_line = ""

        for word in tk_line:
            new_line += word + " "

        output.append(new_line)

    return output


def get_text_cloud(tokens):
    """
    faz a nuvem
    """

    text = ""

    for tk_line in tokens:

        for word in tk_line:
            text += word + " "

    return text


def get_freq_dist_list(tokens):
    """
    prepara os tokens para obter a lista de frequencia das palavras
    usando FreqDist

    from nltk.probability import FreqDist
    """
    output = []

    for tk_line in tokens:
        for word in tk_line:
            output.append(word)

    return output
