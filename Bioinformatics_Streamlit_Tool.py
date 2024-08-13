#Library Imports
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle as pkl
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind
import scanpy as sc
import gseapy as gp
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import load_example_data

st.set_page_config(layout="wide")

st.title("Bioinformatics Tool")
st.write("This simple web-app supports researchers by providing a seamless method to conduct key bioinformatics analyses (DEA & GSEA) on relevant datasets.")


#make a 'cache' dir and a 'gene_sets' dir if they dont already exist
if not os.path.exists('./cache'):
    os.makedirs('./cache')
if not os.path.exists('./gene_sets'):
    os.makedirs('./gene_sets')


#Loading Files Function
@st.cache_data()
def load_data(file):
    return pd.read_csv(file)

#DEA Function
def make_dds(df_counts, metadata):
    inference = DefaultInference(n_cpus=-1)
    dds = DeseqDataSet(
        counts=df_counts,
        metadata=metadata,
        design_factors="CANCER_TYPE_DETAILED",  # compare samples based on the "condition"
        refit_cooks=True,
        inference=inference,
        quiet=True
    )

    dds.fit_size_factors()
    #dds.obsm["size_factors"]

    dds.fit_genewise_dispersions()
    #dds.varm["genewise_dispersions"]

    dds.fit_dispersion_trend()
    #dds.uns["trend_coeffs"]
    #dds.varm["fitted_dispersions"]

    dds.fit_dispersion_prior()
    print(
        f"logres_prior={dds.uns['_squared_logres']}, sigma_prior={dds.uns['prior_disp_var']}"
    )

    dds.fit_MAP_dispersions()
    #dds.varm["MAP_dispersions"]
    #dds.varm["dispersions"]

    dds.fit_LFC()
    #dds.varm["LFC"]

    dds.calculate_cooks()
    if dds.refit_cooks:
        # Replace outlier counts
        dds.refit()

    return dds

#GSEA Function 1- Preranking
def preranked_gsea(genes_df: pd.DataFrame,
                   threshold: int,
                 by_column: str) -> pd.DataFrame:
    genes_df = genes_df.sort_values(by=by_column, ascending=False)
    genes_df = genes_df.head(threshold)
    return genes_df

def fullgenesets_scatter(gsea_df: gp.prerank,
                 plot_n_genesets: int,
                 title: str) -> None:

    gsea_df = gsea_df.res2d.sort_index().head(plot_n_genesets)
    gsea_df.to_csv(f'./gene_sets/{title}.csv')

    gsea_df = gsea_df.sort_values(by=['NES'])
    gsea_df['sig_at_fdr10'] = gsea_df['FDR q-val'] < 0.1
    edge_widths = np.where(gsea_df['sig_at_fdr10'], 3, 0.5)

    fig, ax = plt.subplots(figsize=(4, len(gsea_df) / 1.7))
    scatterplot = ax.scatter(y=gsea_df['Term'], x=gsea_df['NES'], c=gsea_df['FDR q-val'], cmap='PiYG',
                             linewidths=edge_widths, edgecolors='black', s=200)

    cbar = plt.colorbar(scatterplot, orientation="horizontal", pad=1 / len(gsea_df))
    cbar.set_label("False Discovery Rate")
    plt.vlines(0, 0, len(gsea_df) - 1, color='black', linestyles='--', linewidth=0.2)
    plt.title(title, size=12)
    plt.xlabel('Enrichment Score')

    st.pyplot(fig)

#GSEA Function 2-
def calculate_and_plot_genesets(raw_df: pd.DataFrame,
                                n_top_genes: int,
                                plot_n_genesets: int,
                       gene_set) -> None:

    """
    Wrapper function to calculate and plot gene sets
    :param raw_df: input, should be a list of genes (hugo symbol) and their model weights
    :param n_top_genes: integer to specify how many top genes to use for GSEA
    :param plot_n_genesets: integer to specify how many genesets to plot
    :param gene_set: api call to gene set, find the list of gene sets with gp.get_library_name()
    :return: None
    """

    preranked_df = preranked_gsea(genes_df=raw_df,
                                  threshold=n_top_genes,
                                 by_column='padj')

    print("Calculating and plotting GSEA for gene set: ", gene_set)

    pre_res = gp.prerank(rnk=preranked_df,
                         gene_sets=gene_set,
                         processes=4,
                         min_size=5,
                         max_size=1000,
                         permutation_num=200,
                         outdir=None,
                        verbose=True)

    fullgenesets_scatter(gsea_df= pre_res,
                         plot_n_genesets= plot_n_genesets,
                         title= gene_set)
#Regex Search Function
def regex_search(df, search_list):
    pattern = r'\b(?:' + '|'.join(search_list) + r')\b'
    filtered_df = df[df['Term'].str.contains(pattern, regex=True, na=False)]
    print(pattern)
    return filtered_df

#Split Data Import, DEA & GSEA into 3 tabs:
tab1, tab2, tab3 = st.tabs(['Data Processing', 'DEA', 'GSEA'])


with tab1:
    st.subheader('Upload your data')
    col1, col2, col3 = st.columns(3)
    #Uploading Data
    with col1:
        rnaseq_file = st.file_uploader("Select RNA-Seq data to import (CSV file only)", type="csv")
        if rnaseq_file is not None:
            rnaseq_data = load_data(rnaseq_file)
            st.write('**Your imported RNA-seq Data**')
            #only write the first 20 cols and 10 rows for memory efficiency
            st.write(rnaseq_data.iloc[:10, :20])

    with col2:
        clinical_file = st.file_uploader("Select clinical data to import (CSV file only)", type="csv")
        if clinical_file is not None:
            clinical_data = load_data(clinical_file)
            st.write('**Your imported Clinical Data**')
            st.write(clinical_data.head(10))

    with col3:
        label_file = st.file_uploader("Select labels data to import (CSV file only)", type="csv")
        if label_file is not None:
            label_data = load_data(label_file)
            st.write('**Your imported Label Data**')
            st.write(label_data.head(10))
            process_data = True



            # Enabling further processing only if data is available

        else:
            process_data = False
            #st.warning("Please upload all required files to proceed with analysis.")

    if process_data:
        with st.expander("Filter data based on T, N, M Stages"):
            st.success("All files uploaded successfully!")
            # Merging Clinical and Labels Datasets
            df_clinical_label = pd.merge(clinical_data, label_data, on="SAMPLE_ID")
            st.session_state['df_clinical_label'] = df_clinical_label
            st.session_state['rnaseq_data'] = rnaseq_data
            st.session_state['clinical_data'] = clinical_data
            st.session_state['label_data'] = label_data


            t_stage_options = df_clinical_label['PATH_T_STAGE'].unique().tolist()
            n_stage_options = df_clinical_label['PATH_N_STAGE'].unique().tolist()
            m_stage_options = df_clinical_label['PATH_M_STAGE'].unique().tolist()

            selected_t_stage = st.multiselect("Select PATH_T_STAGE", t_stage_options)
            selected_n_stage = st.multiselect("Select PATH_N_STAGE", n_stage_options)
            selected_m_stage = st.multiselect("Select PATH_M_STAGE", m_stage_options)

            if selected_t_stage and selected_n_stage and selected_m_stage:
                df_tnm_filter = df_clinical_label[
                    (df_clinical_label['PATH_T_STAGE'].isin(selected_t_stage)) &
                    (df_clinical_label['PATH_N_STAGE'].isin(selected_n_stage)) &
                    (df_clinical_label['PATH_M_STAGE'].isin(selected_m_stage))
                    ]
                st.write(df_tnm_filter)
                st.write(df_tnm_filter.value_counts('CANCER_TYPE_DETAILED'))

                # Randomly Sample Data to Provide Balanced Analysis
                min_count = df_tnm_filter['CANCER_TYPE_DETAILED'].value_counts().min()
                df_randomsample = df_tnm_filter.groupby('CANCER_TYPE_DETAILED').sample(n=min_count, random_state=52)
                df_sample = df_randomsample.sort_index(ascending=True)
                st.write("Randomly sampled data for balanced analysis")
                st.write(df_sample)

                # Preparing data for analysis
                sample_ids = df_sample['SAMPLE_ID'].unique()
                df_counts = rnaseq_data[rnaseq_data['SAMPLE_ID'].isin(sample_ids)]
                metadata = label_data[label_data['SAMPLE_ID'].isin(sample_ids)]

                df_counts.index = df_counts['SAMPLE_ID']
                df_counts = df_counts.drop(['SAMPLE_ID'], axis=1)
                df_counts = df_counts.astype(int)

                metadata.index = metadata['SAMPLE_ID']
                metadata = metadata.drop(['SAMPLE_ID'], axis=1)

                st.write(
                    "Congrats! You now have your Counts Data & Metadata ready for input to Differential Gene Expression Analysis (DEA).")
                st.write('**Counts Data:**')
                st.write(df_counts)
                st.write('**Metadata:**')
                st.write(metadata)

                # Saving processed data
                df_counts.to_csv('./cache/current_dfcounts.csv')
                metadata.to_csv('./cache/current_metadata.csv')

with tab2:
    st.subheader('Differential Gene Expression Analysis (DEA)')
    countfilepath = os.path.normpath('./cache/current_dfcounts.csv')
    metafilepath = os.path.normpath('./cache/current_metadata.csv')
    if os.path.exists(countfilepath):
        st.success("Files processed successfully. Ready to proceed.")
    else:
        st.warning("Files not processed successfully. Have you run Tab 1?")
    #DGEA using PyDeseq2
    if st.button(label='Run PyDESEQ2 to complete DGEA', type="primary"):
        with st.spinner('Running PyDESeq2 Analysis...'):

                mydds = make_dds(df_counts,metadata)
                st.session_state.mydds = mydds

                stat_res = DeseqStats(mydds, alpha=0.05, cooks_filter=True, independent_filter=True, quiet=True)
                stat_res.run_wald_test()
                # stat_res.p_values

                if stat_res.cooks_filter:
                    stat_res._cooks_filtering()
                # stat_res.p_values

                if stat_res.independent_filter:
                    stat_res._independent_filtering()
                else:
                    stat_res._p_value_adjustment()

                #stat_res.padj
                stat_res.summary()
                df_deseq = stat_res.results_df
                st.session_state.df_deseq = df_deseq
                df_deseq.rename_axis('genes', inplace=True)
                st.write("**Your Differential Gene Expression Analysis has successfully completed!**")
                df_deseq.to_csv('./cache/current_dfdeseq.csv')
    else:
        st.warning('Have you run PyDESeq2? You will need to do this first before visualisation of DEGs.')


    #Options for Data Visualisation of DEGs
    st.write("*Please select how you would like to visualise DEGs from the sidebar.*")
    option = st.sidebar.radio("Select DEG Visualisation", ("Volcano Plot", "Heatmap", "PCA plot"))

    if st.sidebar.button('Submit'):
        if 'df_deseq' in st.session_state and 'mydds' in st.session_state:
            df_deseq = st.session_state.df_deseq
            mydds = st.session_state.mydds
            col1,col2=st.columns(2)
            if option == "Volcano Plot":
                st.write("Your Volcano Plot")
                padj_threshold = 0.05
                log2fc_threshold = 1
                colours = np.where(
                    (df_deseq['padj'] < padj_threshold) & (df_deseq['log2FoldChange'] > log2fc_threshold), 'red',
                    np.where(
                        (df_deseq['padj'] < padj_threshold) & (df_deseq['log2FoldChange'] < -log2fc_threshold),
                        'blue',
                        'gray'
                    )
                )
                fig, ax = plt.subplots(figsize=(8, 6))
                sc = ax.scatter(df_deseq['log2FoldChange'], -np.log10(df_deseq['padj']), c=colours, alpha=0.5)
                ax.set_xlabel('Log2 Fold Change')
                ax.set_ylabel('-log10(adjusted p-value)')
                ax.set_title('Volcano Plot')
                ax.axhline(y=-np.log10(0.05), color='r', linestyle='--', linewidth=1.5)
                ax.axvline(x=1, color='b', linestyle='--', linewidth=1.5)
                ax.axvline(x=-1, color='b', linestyle='--', linewidth=1.5)
                ax.set_xlim(-6, 6)
                ax.set_ylim(0, 20)
                with col1:
                    st.pyplot(fig)

            elif option == "Heatmap":
                st.write("Your Heatmap")
                top_genes = list(df_deseq.sort_values('padj').head(50).index)
                counts_df_top = df_counts[top_genes]
                counts_df_top = counts_df_top.T
                counts_df_top = np.log1p(counts_df_top)
                g = sns.clustermap(counts_df_top, cmap='inferno', z_score=0)
                st.pyplot(g.fig)

            elif option == "PCA plot":
                st.write("Your PCA plot")
                mydds.deseq2()
                sc.tl.pca(mydds)
                fig, ax = plt.subplots()
                sc.pl.pca(mydds, color = 'CANCER-TYPE-DETAILED', size = 200, ax=ax, show=False)
                st.pyplot(fig)

with tab3:
    st.subheader('Gene Set Enrichment Analysis (GSEA)')

    #Checking if DEGs have been imported
    deseqfilepath = os.path.normpath('./cache/current_dfdeseq.csv')
    if os.path.exists(deseqfilepath):
        st.success("Files processed successfully. Ready to proceed.")
    else:
        st.warning("Files not processed successfully. Have you run DEA?")

    df_deseq = load_data(deseqfilepath)
    st.session_state.df_deseq = df_deseq

    #Preparing DEGs for GSEA
    my_genes_for_gseapy = df_deseq.sort_values('padj').head(100)
    my_genes_for_gseapy = my_genes_for_gseapy[['genes', 'padj']]
    #my_genes_for_gseapy['gene'] = my_genes_for_gseapy.index
    st.session_state.my_genes_for_gseapy = my_genes_for_gseapy
    my_genes_for_gseapy.to_csv('./cache/current_mygenes.csv')

    if 'my_genes_for_gseapy' in st.session_state:
        my_genes_for_gseapy = st.session_state.my_genes_for_gseapy
        df_gseapy = my_genes_for_gseapy
        df_gseapy['padj'] = -np.log(df_gseapy['padj'])
        df_gseapy.index = df_gseapy['genes']
        df_gseapy = df_gseapy.drop('genes', axis=1)
        st.session_state.df_gseapy = df_gseapy
        st.write('Your Top 100 DEGs for GSEA:')
        st.write(df_gseapy)

        df_gseapy.to_csv('./cache/current_dfgseapy.csv')

        gene_set_libraries = gp.get_library_name()

        select_gsea_libraries = st.multiselect("Choose Gene Set Libraries for GSEA:",
                                               options=gene_set_libraries)

        if st.button("Run GSEA"):
            if select_gsea_libraries:
                st.write(f"You have chosen:{select_gsea_libraries}")
            else:
                st.warning("Please select at least one Gene Set Library to run GSEA.")

            gene_sets = select_gsea_libraries
            for gene_set in gene_sets:
                calculate_and_plot_genesets(df_gseapy,
                                            1000,
                                            50,
                                            f'{gene_set}')