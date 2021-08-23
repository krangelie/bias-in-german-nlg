import random

from src import constants


def replace_with_gendered_pronouns(augment, text_col, df):
    assert len(set(df.Gender.unique()) - set(["F", "M", "N"])) == 0
    if augment == "single_gender":
        df = replace_with_single_option(text_col, df)
    elif augment == "list_gender":
        df = replace_from_list(text_col, df)
    else:
        SystemExit("Asking for non-specified augmentation option")

    return df


def replace_from_list(text_col, df):
    # For all sentences with female indication, prepend female pronoun/ subject
    df.loc[df["Gender"] == "F", text_col] = df.loc[df["Gender"] == "F", text_col].apply(
        lambda text: text.replace(
            "Die Person",
            random.choice(constants.FEMALE_LIST),
        )
    )

    print(df.loc[df["Gender"] == "F", text_col][:5])

    # For all sentences with male indication, prepend male pronoun/ subject
    df.loc[df["Gender"] == "M", text_col] = df.loc[df["Gender"] == "M", text_col].apply(
        lambda text: text.replace(
            "Die Person",
            random.choice(constants.MALE_LIST),
        )
    )

    print(df.loc[df["Gender"] == "M", text_col][:5])

    # For all sentences without any gender indication, gender randomly
    df.loc[df["Gender"] == "N", text_col] = df.loc[df["Gender"] == "N", text_col].apply(
        lambda text: text.replace(
            "Die Person",
            random.choice(
                [
                    random.choice(constants.FEMALE_LIST),
                    random.choice(constants.MALE_LIST),
                ]
            ),
        )
    )

    print(df.loc[df["Gender"] == "N", text_col][:20])

    return df


def replace_with_single_option(text_col, df):
    # For all sentences with female indication, prepend female pronoun/ subject
    df.loc[df["Gender"] == "F", text_col] = df.loc[
        df["Gender"] == "F", text_col
    ].str.replace("Die Person", constants.FEMALE_SINGLE)

    print(df.loc[df["Gender"] == "F", text_col][:5])

    # For all sentences with male indication, prepend male pronoun/ subject
    df.loc[df["Gender"] == "M", text_col] = df.loc[
        df["Gender"] == "M", text_col
    ].str.replace("Die Person", constants.MALE_SINGLE)

    print(df.loc[df["Gender"] == "M", text_col][:5])

    # For all sentences without any gender indication, gender randomly
    df.loc[df["Gender"] == "N", text_col] = df.loc[df["Gender"] == "N", text_col].apply(
        lambda text: text.replace(
            "Die Person",
            random.choice([constants.FEMALE_SINGLE, constants.MALE_SINGLE]),
        )
    )

    print(df.loc[df["Gender"] == "N", text_col][:20])

    return df
