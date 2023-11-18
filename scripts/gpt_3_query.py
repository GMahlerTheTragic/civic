from openai import OpenAI
import pandas as pd

# from config import PROJECT_ROOT
# import os
#
client = OpenAI()
#
#
# def get_gpt_4_example(prependString, abstract):
#     return f"""METADATA: \"{prependString}\"\n ABSTRACT: \"{abstract}\""""
#
#
# def get_random_samples(N):
#     df = pd.read_csv(
#         os.path.join(PROJECT_ROOT, "data/02_processed/civic_evidence_train.csv")
#     )
#     sampled_df = pd.DataFrame()
#
#     # Iterate over unique values in the 'evidence_level' column
#     for value in df["evidenceLevel"].unique():
#         # Select N random samples for each category
#         sampled_rows = df[df["evidenceLevel"] == value].sample(N)
#
#         # Append the sampled rows to the new DataFrame
#         sampled_df = pd.concat([sampled_df, sampled_rows], axis=0)
#     return sampled_df


example_1 = """METADATA: \"DiseaseName: Chronic Myeloid Leukemia \n
Molecular Profile Name: ABL1 BCR::ABL L248V \n
Therapies: Imatinib \n
Significance: RESISTANCE\" \n
ABSTRACT: \"Imatinib-treated chronic myeloid leukemia (CML) patients with acquired resistance commonly 
have detectable BCR-ABL kinase domain mutations. It is unclear whether patients who remain sensitive to imatinib also 
have a significant incidence of mutations. We evaluated 144 patients treated with imatinib for BCR-ABL kinase domain 
mutations by direct sequencing of 40 accelerated phase (AP), 64 late chronic phase (> or = 12 months from diagnosis, 
late-CP), and 40 early-CP patients. Mutations were detected in 27 patients at 17 different residues, 13 (33%) of 40 
in AP, 14 (22%) of 64 in late-CP, and 0 of 40 in early-CP. Acquired resistance was evident in 24 (89%) of 27 patients 
with mutations. Twelve (92%) of 13 patients with mutations in the adenosine triphosphate (ATP) binding loop (P-loop) 
died (median survival of 4.5 months after the mutation was detected). In contrast, only 3 (21%) of 14 patients with 
mutations outside the P-loop died (median follow-up of 11 months). As the detection of mutations was strongly 
associated with imatinib resistance, we analyzed features that predicted for their detection. Patients who commenced 
imatinib more than 4 years from diagnosis had a significantly higher incidence of mutations (18 [41%] of 44) compared 
with those treated within 4 years (9 [9%] of 100), P <.0001. Lack of a major cytogenetic response (MCR) was also 
associated with a higher likelihood of detecting a mutation; 19 (38%) of 50 patients without a MCR had mutations 
compared with 8 (8.5%) of 94 with an MCR, P <.0001. In conclusion, the detection of kinase domain mutations using a 
direct sequencing technique was almost always associated with imatinib resistance, and patients with mutations in the 
P-loop had a particularly poor prognosis.\""""

label_1 = "C"

example_2 = """METADATA: \"DiseaseName: Breast Cancer \n Molecular Profile Name: ERBB2 Amplification \n Therapies: 
Lapatinib-Trastuzumab \n Significance: SENSITIVITYRESPONSE\" \n ABSTRACT: \"Antibodies and small molecule tyrosine 
kinase inhibitors targeting ErbB2 exhibit distinct, noncross resistant mechanisms of action. Here, apoptosis of 
ErbB2-overexpressing breast cancer cells was enhanced by combining lapatinib, an inhibitor of ErbB1 and ErbB2 
tyrosine kinases, with anti-ErbB2 antibodies, including (i) trastuzumab, a humanized monoclonal antibody, 
and (ii) pAb, rabbit polyclonal antisera generated by vaccination with a human ErbB2 fusion protein. Treating 
ErbB2-overexpressing breast cancer cell lines with a relatively low concentration of lapatinib alone resulted in a 
minimal increase in tumor cell apoptosis with an associated decrease in steady-state protein levels of p-ErbB2, 
p-Akt, p-Erk1/2, and notably survivin, compared to baseline. Exposure to pAb alone reduced total ErbB2 protein, 
disrupting ErbB3 transactivation, leading to a marked inhibition of p-Akt; however, survivin protein levels remained 
unchanged and apoptosis only increased slightly. Treatment with trastuzumab alone had relatively little effect on 
survivin and apoptosis was unaffected. Combining lapatinib with either pAb or trastuzumab markedly downregulated 
survivin protein and enhanced tumor cell apoptosis. The association between the inhibition of survivin and enhanced 
apoptosis following the combination of ErbB2-targeted therapies provides a biological effect in order to identify 
therapeutic strategies that promote tumor cell apoptosis and might improve clinical response.\""""

label_2 = "D"

example_3 = """METADATA: \"DiseaseName: NUT Midline Carcinoma \n Molecular Profile Name: BRD4 BRD4::NUTM1 \n 
Therapies: Unknown \n Significance: POSITIVE\" \n ABSTRACT: \"Undifferentiated carcinomas of the upper aerodigestive 
tract (UCUAT) occur most frequently within the nasopharynx and are most often associated with infection by 
Epstein-Barr virus (EBV) (WHO undifferentiated nonkeratinizing squamous cell carcinoma). An unusual group of 
aggressive carcinomas are characterized by translocations that involve Nuclear Protein in Testis (NUT), a novel gene 
on chromosome 15. In about two-thirds of cases, NUT is fused to BRD4 on chromosome 19. These tumors, here termed NUT 
midline carcinomas (NMCs), are undifferentiated, may have focal squamous differentiation, and are reported to occur 
in children and young adults. This study investigates the prevalence of NUT rearrangement and the diagnostic 
significance of NUT expression in a series of upper aerodigestive tract undifferentiated carcinomas. The histologic 
features of these tumors are described in detail.All UCUAT not associated with EBV infection seen at the University 
of Virginia (UVA) over a 16-year period were reviewed. Clinical and histologic features were noted. Additional 
material was submitted for fluorescent in situ hybridization (FISH) using split-apart probes to the NUT and BRD4 
genes. Immunohistochemistry (IHC) was performed on all cases using a polyclonal antibody to NUT, and on select cases 
with antibody to p63.Thirty-one UCUAT were identified. Twenty-five tumors had originally been diagnosed as sinonasal 
undifferentiated carcinomas. Five of 28 cases (2 males, 3 females; average age 47; range 31 to 78) with interpretable 
results showed rearrangements of the NUT and BRD4 genes by FISH. Three of these 5 cases showed diffuse (>90%) nuclear 
staining for NUT by IHC; 22 of 23 other tumors showed at most focal (<50%) nuclear staining. Undifferentiated 
carcinomas with NUT gene rearrangement had focal abrupt squamous differentiation in 2 cases, and intense and diffuse 
immunoreactivity with antibody to p63 in 4 cases.Approximately, 20% of UCUAT not associated with EBV infection were 
found to have rearrangements of NUT by FISH. Although previous reports suggest that NMCs afflict only children and 
young adults, 4 of 5 of the patients described are mature adults older than any heretofore reported, suggesting that 
previous reports may have been biased in their case selections. Furthermore, because these tumors are 
indistinguishable from other poorly differentiated carcinomas, IHC using NUT antibody may be a useful method for the 
identification of these tumors. Despite the lack of overt squamous differentiation in most cases, their p63 
immunoreactivity suggests that NMCs may generally be of squamous lineage.\""""

label_3 = "B"

example_4 = """METADATA: \"DiseaseName: Her2-receptor Negative Breast Cancer \n Molecular Profile Name: BRCA2 
Loss-of-function \n Therapies: Olaparib \n Significance: SENSITIVITYRESPONSE\" \n ABSTRACT: \"Poly(adenosine 
diphosphate-ribose) polymerase inhibitors target cancers with defects in homologous recombination repair by synthetic 
lethality. New therapies are needed to reduce recurrence in patients with BRCA1 or BRCA2 germline mutation-associated 
early breast cancer.We conducted a phase 3, double-blind, randomized trial involving patients with human epidermal 
growth factor receptor 2 (HER2)-negative early breast cancer with BRCA1 or BRCA2 germline pathogenic or likely 
pathogenic variants and high-risk clinicopathological factors who had received local treatment and neoadjuvant or 
adjuvant chemotherapy. Patients were randomly assigned (in a 1:1 ratio) to 1 year of oral olaparib or placebo. The 
primary end point was invasive disease-free survival.A total of 1836 patients underwent randomization. At a 
prespecified event-driven interim analysis with a median follow-up of 2.5 years, the 3-year invasive disease-free 
survival was 85.9% in the olaparib group and 77.1% in the placebo group (difference, 8.8 percentage points; 95% 
confidence interval [CI], 4.5 to 13.0; hazard ratio for invasive disease or death, 0.58; 99.5% CI, 0.41 to 0.82; 
P<0.001). The 3-year distant disease-free survival was 87.5% in the olaparib group and 80.4% in the placebo group (
difference, 7.1 percentage points; 95% CI, 3.0 to 11.1; hazard ratio for distant disease or death, 0.57; 99.5% CI, 
0.39 to 0.83; P<0.001). Olaparib was associated with fewer deaths than placebo (59 and 86, respectively) (hazard 
ratio, 0.68; 99% CI, 0.44 to 1.05; P = 0.02); however, the between-group difference was not significant at an 
interim-analysis boundary of a P value of less than 0.01. Safety data were consistent with known side effects of 
olaparib, with no excess serious adverse events or adverse events of special interest.Among patients with high-risk, 
HER2-negative early breast cancer and germline BRCA1 or BRCA2 pathogenic or likely pathogenic variants, 
adjuvant olaparib after completion of local treatment and neoadjuvant or adjuvant chemotherapy was associated with 
significantly longer survival free of invasive or distant disease than was placebo. Olaparib had limited effects on 
global patient-reported quality of life. (Funded by the National Cancer Institute and AstraZeneca; OlympiA 
ClinicalTrials.gov number, NCT02032823.).\""""

label_4 = "A"

example_5 = """METADATA: \"DiseaseName: Melanoma \n Molecular Profile Name: NTRK1 GON4L::NTRK1 \n Therapies: 
Larotrectinib \n Significance: SENSITIVITYRESPONSE\" \n ABSTRACT: \"A number of oncogenic driver mutations have been 
identified in melanocytic nevi and melanoma, but translocations also play a role in tumorigenesis and provide 
potential therapeutic targets for malignant lesions. Various translocations, such as those involving the anaplastic 
lymphoma kinase (ALK), neurotrophic tropomyosin receptor kinase 1 (NTRK1), and NTRK3 have been reported in spitzoid 
melanocytic neoplasms leading to kinase-fusion proteins that result in immunohistochemically detectable ALK or NTRK 
expression. We have previously reported that ALK expression can be found in nonspitzoid primary and metastatic 
cutaneous melanomas. In this study we report that nonspitzoid metastasizing melanomas of adults may also harbor NTRK 
fusions and that NTRK expression can be immunohistochemically detected in these tumors. Of 751 melanomas analyzed by 
next-generation sequencing, 4 metastatic melanomas were identified with NTRK fusions, 3 involving NTRK1, 1 involving 
NTRK2. They occurred in 3 women and 1 man. Two of the corresponding primary tumors were from the trunk, 
1 from an extremity and 1 tumor arose in anal skin. One primary tumor displayed features of superficial spreading 
melanoma and 3 were nodular melanomas. All tumors were cytologically characterized by the presence of large 
epithelioid melanocytes. All tumors were immunoreactive with anti-Trk antibody. Next-generation sequencing documented 
that the NTRK1 fusion partners included TRIM63, DDR2, and GON4L. One tumor harbored an NTRK2-TRAF2 fusion. Thus, 
our findings document that NTRK kinase fusions can occur in nonspitzoid metastasizing melanomas of adults. The 
presence of an NTRK family fusion in these tumors may provide a therapeutic opportunity in a small subset of patients 
with metastatic melanoma.\""""

label_5 = "E"

example_6 = """METADATA: \"DiseaseName: Acute Myeloid Leukemia \n Molecular Profile Name: FLT3 Mutation \n Therapies: 
Gilteritinib \n Significance: SENSITIVITYRESPONSE\" \n ABSTRACT: \"Patients with relapsed or refractory acute myeloid 
leukemia (AML) with mutations in the FMS-like tyrosine kinase 3 gene (FLT3) infrequently have a response to salvage 
chemotherapy. Gilteritinib is an oral, potent, selective FLT3 inhibitor with single-agent activity in relapsed or 
refractory FLT3-mutated AML.In a phase 3 trial, we randomly assigned adults with relapsed or refractory FLT3-mutated 
AML in a 2:1 ratio to receive either gilteritinib (at a dose of 120 mg per day) or salvage chemotherapy. The two 
primary end points were overall survival and the percentage of patients who had complete remission with full or 
partial hematologic recovery. Secondary end points included event-free survival (freedom from treatment failure [
i.e., relapse or lack of remission] or death) and the percentage of patients who had complete remission.Of 371 
eligible patients, 247 were randomly assigned to the gilteritinib group and 124 to the salvage chemotherapy group. 
The median overall survival in the gilteritinib group was significantly longer than that in the chemotherapy group (
9.3 months vs. 5.6 months; hazard ratio for death, 0.64; 95% confidence interval [CI], 0.49 to 0.83; P<0.001). The 
median event-free survival was 2.8 months in the gilteritinib group and 0.7 months in the chemotherapy group (hazard 
ratio for treatment failure or death, 0.79; 95% CI, 0.58 to 1.09). The percentage of patients who had complete 
remission with full or partial hematologic recovery was 34.0% in the gilteritinib group and 15.3% in the chemotherapy 
group (risk difference, 18.6 percentage points; 95% CI, 9.8 to 27.4); the percentages with complete remission were 
21.1% and 10.5%, respectively (risk difference, 10.6 percentage points; 95% CI, 2.8 to 18.4). In an analysis that was 
adjusted for therapy duration, adverse events of grade 3 or higher and serious adverse events occurred less 
frequently in the gilteritinib group than in the chemotherapy group; the most common adverse events of grade 3 or 
higher in the gilteritinib group were febrile neutropenia (45.9%), anemia (40.7%), and thrombocytopenia (
22.8%).Gilteritinib resulted in significantly longer survival and higher percentages of patients with remission than 
salvage chemotherapy among patients with relapsed or refractory FLT3-mutated AML. (Funded by Astellas Pharma; ADMIRAL 
ClinicalTrials.gov number, NCT02421939.).\""""

label_6 = "B"


def main():
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": """You are an expert on rare tumor treatments. In the following you will be presented with 
                abstracts from medical research papers. These abstracts deal with treatment approaches for rare 
                cancers as characterized by their specific genomic variants. Your job is to infer the level of 
                clinical significance of the investigations described in the abstract from the abstract and relevant 
                metadata. The labels should range from \"A\" (indicating the strongest clinical significance) to 
                \"E\" (indicating the weakest clinical significance). You will answer machine-like with exactly one 
                character (the level of clinical significance you think is appropriate). You will be presented with 
                examples.""",
            },
            {
                "role": "user",
                "content": f"""Extract the level of clinical significance from this combination of metadata and 
                abstract: \n {example_1}""",
            },
            {
                "role": "assistant",
                "content": f"""{label_1}""",
            },
            {
                "role": "user",
                "content": f"""Extract the level of clinical significance from this combination of metadata and 
                    abstract: \n {example_2}""",
            },
            {
                "role": "assistant",
                "content": f"""{label_2}""",
            },
            {
                "role": "user",
                "content": f"""Extract the level of clinical significance from this combination of metadata and 
                    abstract: \n {example_3}""",
            },
            {
                "role": "assistant",
                "content": f"""{label_3}""",
            },
            {
                "role": "user",
                "content": f"""Extract the level of clinical significance from this combination of metadata and 
                    abstract: \n {example_4}""",
            },
            {
                "role": "assistant",
                "content": f"""{label_4}""",
            },
            {
                "role": "user",
                "content": f"""Extract the level of clinical significance from this combination of metadata and 
                    abstract: \n {example_5}""",
            },
            {
                "role": "assistant",
                "content": f"""{label_5}""",
            },
            {
                "role": "user",
                "content": f"""Extract the level of clinical significance from this combination of metadata and 
                        abstract: \n {example_6}""",
            },
        ],
    )

    print(completion.choices[0].message)


if __name__ == "__main__":
    main()
