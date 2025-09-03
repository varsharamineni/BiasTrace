import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_by_social_value(results):
    """Aggregate incorrect-and-stereotype counts by Relevant_social_values."""
    df = pd.DataFrame(results)
    df['Relevant_social_values'] = df['Relevant_social_values'].fillna('Unknown').str.strip()

    summary = df.groupby('Relevant_social_values').agg(
        n_disambig_total=('ambiguous', lambda x: (~x).sum()),
        n_ambig_total=('ambiguous', 'sum'),
        n_incorrect_and_stereotype_disambig=('incorrect_and_stereotype', lambda x: ((x) & (~df.loc[x.index, 'ambiguous'])).sum()),
        n_incorrect_and_stereotype_ambig=('incorrect_and_stereotype', lambda x: ((x) & (df.loc[x.index, 'ambiguous'])).sum())
    ).reset_index()

    summary['prop_incorrect_stereo_disambig'] = summary['n_incorrect_and_stereotype_disambig'] / summary['n_disambig_total']
    summary['prop_incorrect_stereo_ambig'] = summary['n_incorrect_and_stereotype_ambig'] / summary['n_ambig_total']

    return summary

def process_all_categories(folder_path):
    all_summaries = []
    for filename in os.listdir(folder_path):
        if filename.endswith('_results_merged.json') and filename.startswith('bbq_'):
            category = filename.split('bbq_')[1].split('_results')[0]
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            results = data['results']
            summary = aggregate_by_social_value(results)
            summary['category'] = category
            all_summaries.append(summary)

    return pd.concat(all_summaries, ignore_index=True)

def plot_incorrect_stereotype(summary_df, output_file=None):
    """Side-by-side bar plot of incorrect & stereotype proportions."""
    summary_long = summary_df.melt(
        id_vars=['Relevant_social_values','category'],
        value_vars=['prop_incorrect_stereo_disambig','prop_incorrect_stereo_ambig'],
        var_name='context',
        value_name='proportion'
    )

    # Make nicer labels
    summary_long['context'] = summary_long['context'].replace({
        'prop_incorrect_stereo_disambig': 'Disambiguated',
        'prop_incorrect_stereo_ambig': 'Ambiguous'
    })

    plt.figure(figsize=(12,6))
    sns.barplot(
        data=summary_long,
        x='Relevant_social_values',
        y='proportion',
        hue='context'
    )
    plt.ylabel('Proportion of Incorrect & Stereotype')
    plt.xlabel('Relevant Social Value')
    plt.title('Incorrect & Stereotype Predictions by Social Value and Context')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Context')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    plt.show()

def plot_incorrect_stereotype_by_category(summary_df, output_file=None):
    """Separate plots for each category showing incorrect & stereotype proportions."""
    categories = summary_df['category'].unique()
    n_categories = len(categories)
    
    ncols = 2
    nrows = (n_categories + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*6, nrows*5), squeeze=False)
    
    for idx, category in enumerate(categories):
        ax = axes[idx // ncols, idx % ncols]
        cat_df = summary_df[summary_df['category'] == category]
        cat_long = cat_df.melt(
            id_vars=['Relevant_social_values'],
            value_vars=['prop_incorrect_stereo_disambig','prop_incorrect_stereo_ambig'],
            var_name='context',
            value_name='proportion'
        )
        cat_long['context'] = cat_long['context'].replace({
            'prop_incorrect_stereo_disambig': 'Disambiguated',
            'prop_incorrect_stereo_ambig': 'Ambiguous'
        })

        sns.barplot(
            data=cat_long,
            x='Relevant_social_values',
            y='proportion',
            hue='context',
            ax=ax
        )
        ax.set_title(f'Category: {category}')
        ax.set_ylabel('Proportion Incorrect & Stereotype')
        ax.set_xlabel('Relevant Social Value')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.legend(title='Context')

    # Hide any empty subplots
    for j in range(idx+1, nrows*ncols):
        axes[j // ncols, j % ncols].axis('off')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    plt.show()


if __name__ == "__main__":
    folder_path = 'outputs/qwen_full_8B_simple_prompt/20250827_163953'
    summary_df = process_all_categories(folder_path)

    # Separate plots per category
    plot_incorrect_stereotype_by_category(
        summary_df, 
        output_file=os.path.join(folder_path, 'incorrect_stereotype_by_social_value_per_category.png')
    )