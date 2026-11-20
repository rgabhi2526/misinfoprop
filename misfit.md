|                        | |                    |     |     | |                      |     |     |     |     |     |     |
| ---------------------- | -------------------- | --- | --- | ---------------------- | --- | --- | --- | --- | --- | --- |
| Received:28October2020 | Revised:8January2021 |     |     | Accepted:29January2021 |     |     |     |     |     |     |
DOI:10.1002/int.22384
| RESEARCH ARTICLE |           |             |             |            |            |     |         |        |     |     |
| ---------------- | --------- | ----------- | ----------- | ---------- | ---------- | --- | ------- | ------ | --- | --- |
| Intelligent      | deception |             |             | techniques |            |     | against |        |     |     |
| adversarial      | attack    |             | on          | the        | industrial |     |         | system |     |     |
| Kumari1          |           |             |             |            |            |     | Yadav2  |        |     |     |
| Suchi            | |         | Riteshkumar |             |            | Jayprakash |     |         |        | |   |     |
| Suyel Namasudra3 |           | |           | Ching‐Hsien |            | Hsu4,5,6   |     |         |        |     |     |
1DepartmentofComputerScience
| Engineering,BennettUniversity,Greater |     |     |     | Abstract |     |     |     |     |     |     |
| ------------------------------------- | --- | --- | --- | -------- | --- | --- | --- | --- | --- | --- |
Noida,UtterPradesh,India
Communitydetectionalgorithms(CDAs)areaimingto
2VivekanandEducationSociety's
|     |     |     |     | group | nodes based | on  | their connections |     | and | play an |
| --- | --- | --- | --- | ----- | ----------- | --- | ----------------- | --- | --- | ------- |
InstituteofTechnology,Mumbai,
essentialroleinthecomplexsystemanalysis.However,
Maharashtra,India
3DepartmentofComputerScienceand for privacy reasons, we may want to prevent commu-
Engineering,NationalInstituteof nities or a group of nodes in the complex industrial
TechnologyPatna,Bihar,India
|     |     |     |     | network | from | being | discovered | in  | some instances, |     |
| --- | --- | --- | --- | ------- | ---- | ----- | ---------- | --- | --------------- | --- |
4SchoolofMathematicsandBigData,
|     |     |     |     | leading | to the topics | on  | community |     | deception. | In this |
| --- | --- | --- | --- | ------- | ------------- | --- | --------- | --- | ---------- | ------- |
FoshanUniversity,Foshan,China
5DepartmentofComputerScienceand paper, we introduce and formalize two intelligent
InformationEngineering,Asia community deception methods to conceal the nodes
| University,Taichung,Taiwan |     |     |     |              |       |     |         | node‐based |           |     |
| -------------------------- | --- | --- | --- | ------------ | ----- | --- | ------- | ---------- | --------- | --- |
|                            |     |     |     | from various | CDAs. |     | We used |            | matrices, |     |
6DepartmentofMedicalResearch,China
|     |     |     |     | persistence | and | safeness | scores, |     | to formalize | the |
| --- | --- | --- | --- | ----------- | --- | -------- | ------- | --- | ------------ | --- |
MedicalUniversityHospital,China
MedicalUniversity,Taichung,Taiwan optimization problems to confound the CDAs. The
|     |     |     |     | persistence | score | is used | to  | destabilize | the constant |     |
| --- | --- | --- | --- | ----------- | ----- | ------- | --- | ----------- | ------------ | --- |
Correspondence
communitiesinthenetworkwhilethesafenessscoreis
SuyelNamasudra,Departmentof
ComputerScienceandEngineering, usedtoassessthelevelofhidingofanodefromCDAs.
NationalInstituteofTechnologyPatna,
|     |     |     |     | The objective | functions |     | aim | to minimize | the | persis- |
| --- | --- | --- | --- | ------------- | --------- | --- | --- | ----------- | --- | ------- |
Patna,Bihar,India.
|     |     |     |     | tence score | and | maximize | the | safeness | score | of the |
| --- | --- | --- | --- | ----------- | --- | -------- | --- | -------- | ----- | ------ |
Email:suyelnamasudra@gmail.com
|     |     |     |     | nodes in | the network. |     | From the | simulation | results, | it  |
| --- | --- | --- | --- | -------- | ------------ | --- | -------- | ---------- | -------- | --- |
Fundinginformation
|     |     |     |     | can be | analyzed | that | the proposed |     | strategies | are |
| --- | --- | --- | --- | ------ | -------- | ---- | ------------ | --- | ---------- | --- |
NationalNaturalScienceFoundationof
|     |     |     |     | intelligently | concealing |     | the community |     | information | in  |
| --- | --- | --- | --- | ------------- | ---------- | --- | ------------- | --- | ----------- | --- |
China,Grant/AwardNumber:61872084
|     |     |     |     | the complex | industrial |     | system. |     |     |     |
| --- | --- | --- | --- | ----------- | ---------- | --- | ------- | --- | --- | --- |
KEYWORDS
communitydeception,constantcommunities,persistence,
safeness
|
IntJIntellSyst.2021;1–26. wileyonlinelibrary.com/journal/int ©2021WileyPeriodicalsLLC 1

|
2 KUMARIETAL.
1 | INTRODUCTION
Most of the real‐world systems are complex and can be represented with the help of complex
networks. The networks are composed of nodes/vertices representing the dynamical units of
thesystemandedges/linksrepresentinginteractionsbetweentheunits. Someof theexamples
include terrorist interaction network,1 friendship network,2–4 transportation network,5
protein–protein interaction network,6 and so forth. Real‐world complex networks usually ex-
hibit inhomogeneity in structural and functional properties of nodes, resulting in densely
interconnected nodes, also known as ‘communities.7,8 A community structure is roughly
defined as groups of nodes tightly connected internally and loosely connected externally.
Thecommunitystructureisnotonlyshowingthecloserelationshipamongthenodesinsidethe
community but may also exhibit similar behavior in their respective field.
The community detection algorithms (CDAs) have been designed to unveil some essential
insights,suchashumaninteractionsinsocialnetworks,9,10subjectdescriptionsininformation
networks,11,12 interaction between neurons,13 the association among terrorists in the criminal
network,14,15 and so forth. Community identification tools raise multiple considerable privacy
concerns. Hence, the community deception technique may be used to conceal the associated
community information. The method may be used by security personnel to get information
abouttheterroristassociation.Thesecurityagentsdeploythemselvesinthenetworkassociated
withterroristactivities,whopretendthemselvesasapartoftheterroristcommunitytogetthe
information about the terrorist activities; however, they do not want to be revealed by the
outsideworld.Liketheterroristnetwork,thedeceptionmethodcanbeusedtohidethestatus
of the participants in the social network and so on.11,16,17 Similarly, we may apply deception
technique to hide internet of things (IoT) devices in cloud computing
environment18–24
from
the CDAs. By doing that we can preserve the identity of such devices and it will be helpful to
perform some analysis on such environments.25
Researchers provided multiple CDAs but, Newman provides the first and effective mod-
ularity Q‐based community detection.26 Some of the CDAs are based on the agglomerative
approach, and some are based on divisive approach.27 Some common CDAs are as follows:
Greedy modularity,28 asynchronous label propagation (Asyn LPA),29 K_clique,2 Bisection,30
Girvan–Newman,7Louvain,8andsoon.Mostofthecommunitydetection algorithmsworkon
optimizationproblemandtrytofindtheoptimalpartitionofthenetworkbymaximizingsome
fitness parameter.
Multiple CDAs are available to analyze the network structure but, a little work has been
done to conceal the community information of the nodes in the
network.10,31–34
Researchers
found the motivation behind the hiding of information and discussed various deception me-
chanisms in the social network. Caspi and Gorsky35 performed a web‐based survey on the
motivation and emotions for Israeli users' online deception. It is observed that the frequent
usersdeceivemorethantheinfrequentusers,andtheyounggenerationdeceivesmorethanold
ones. Some users want to hide information only for fun and some users have some wrong
intentionbehindthatdeception.Theconceptofcommunityhidingmechanismwasintroduced
by Nagaraja.31 He studied the effort of the attacker in breaking the community association of
the users via network topology information7 and traffic flow analysis.36 He presented a
modularity‐based community detection technique, and the privacy of the nodes was provided
by using selective surveillance strategies. Waniek et al.10 used the node's neighborhood for
rewiring strategy, and community information is kept hidden by providing random inter-
community connections.Chenetal.37proposedgeneticalgorithm‐basedattack strategywhere

|
KUMARIETAL. 3
fitness function depends on modularity. The approach was tested for seven CDAs on multiple
social networks. Liu et al.38 proposed a community‐based structural entropy to measure the
amount of information revealed through a particular community structured network. They
providedaresidualentropyminimizationformulaforthedeceptionofcommunitystructurein
the network. Li et al.9 aimed to hide targeted nodes from the CDAs using the black‐box
approach.Theyclaimedthatthegeneratedadversarialgraphsthroughtheirproposedapproach
can be transferred to other learning‐based community detection models.
Researchershaveproposedsomemodelstoidentifytheterroristassociationinthecriminal
network. Taha and Yoo14 proposed a tool to find the influential members of the terrorist
organizationthroughtheimmediateneighborsoflower‐levelcriminals.Thiswillhelpanalyze
theflowoftheimportantinformationinthenetwork.TahaandYoo1alsodevelopedaspanning
treeoftheterroristnetworktofindtheleaderofthenetwork.Theleader'sinformationisvery
important to get confidential information and the planned future terrorist activities in the
network. Marshak et al.15 developed a hierarchical evolution of a terrorist network via the
recruitment of people according to their affiliations. At the same time, security agents try to
break the organization through pursuit strategy.
Noneoftheabove‐mentionedresearchworksdiscussedtherewiringmechanismtoconceal
theinformationofthenetworkelements.First,FiondaandPirro39suggestedanedgerewiring‐
basedcommunitydeceptionproblemtohidethetargetcommunityfromCDAs.Theyprovided
a safeness score to each node in the target community and tried to maximize it by diving a
community to more number of communities with less percentage of target nodes. The larger
number of communities may lead to less concealed node information in the network. The
approachesmentionedaboveareprovidingsomestrategiestoconcealcommunityinformation
of the network. The rewiring‐based strategy is, respectively, cheaper than the other strategies
that is why it motivated me to develop a rewiring‐based deception mechanism to conceal the
network information.
1.1 | Motivations and contributions
The world is facing terrorist attacks planned and executed by multiple terrorist organizations,
data breaching in social networks and so on. Hence, it is essential to develop a model to get
insightintothenetworktoprotecttheworldfromsuchkindofattacks.Itisworthmentioning
that due to the high association of the members of the network, they form multiple commu-
nities to accomplish their targets. Therefore, we aim to deploy some security personnel in the
terrorist network or in the social network for getting information about the network. The
identitiesoftheagentsareconcealed sothattheterroristsorindividualscannotsuspectthem.
AnotherperspectiveistoraiseawarenessforthedesignofrobustCDAsagainstsuchdeception
techniques. The main objective of the proposed project is to diminish the community of the
network,concealtheinformationofsecurityagents,anddisintegratethecommunitybasedon
network parameters. The comparisons among the proposed approach and the previous ap-
proaches are given in Table 1. Furthermore, we used other strategies to conceal the node's
identity in the network. The precise contributions of the paper are as follows:
• Wedevelopedapersistence‐basedstrategywherenodesweremovedtothecommunitywith
the most substantial extraction score to destabilize the constant communities (CCs) and to
conceal the nodes' information against CDAs.

|
4
KUMARIETAL.
| TABLE | 1 Comparisonsamongexistingapproaches |     |     |     |     |     |
| ----- | ------------------------------------ | --- | --- | --- | --- | --- |
Effectofdeception
| Approach |     | Updatesconsidered |     | Techniques |     | onCDAs |
| -------- | --- | ----------------- | --- | ---------- | --- | ------ |
Nagaraja31
|                  |     | Addition |     | Modularity                 |     | No  |
| ---------------- | --- | -------- | --- | -------------------------- | --- | --- |
| Wanieketal.10    |     | Addition |     | Modularity                 |     | No  |
| FiondaandPirro39 |     | Rewiring |     | Modularityandsafeness      |     | Yes |
| Proposedapproach |     | Rewiring |     | Modularity,persistence,and |     | Yes |
safeness
Abbreviation:CDA,communitydetectionalgorithm.
•
After that, the safeness score of the selective target agents was maximized with the help of
edge rewiring operation. By doing this, the particular nodes were safeguarded from
community‐centric
attack.
| •   |     |     |     |     |     | real‐world |
| --- | --- | --- | --- | --- | --- | ---------- |
The proposed strategies are tested against multiple CDAs on multiple datasets.
The result is compared with the Fionda and Pirro39 approach and from the simulation
| results, | the performance |     | can be analyzed. |     |     |     |
| -------- | --------------- | --- | ---------------- | --- | --- | --- |
The rest of the paper is organized as follows. In Section 2, some deception metrics are
defined, boundary conditions of some parameters are evaluated, and an overall view of the
model is provided through a running example. We formalized the problem of community
deception andproposed two strategies to conceal the community of the nodes in thenetwork.
The experimental results are analyzed in Section 3. Finally, we conclude the paper and high-
| light future | research | directions | in Section | 4.    |     |     |
| ------------ | -------- | ---------- | ---------- | ----- | --- | --- |
| 2 |          | OVERVIEW | OF         | THE        | MODEL |     |     |
An undirected network, G, can be represented as G = G(,), where  = {i∣1 ≤ i ≤ N},

= {i∣1 ≤ i ≤ E}arethesetofnodesandlinks,respectively,forwhichthenetworkisdefined.
ThetotalnumberofnodesandlinksinthenetworkisrepresentedbyN andE,respectively.A
zero–onematrixAofthesize(N
|     |     |     | × N)isdefinedwhereA |     | i,j = 1,ifnodes,iand | j,areconnected |
| --- | --- | --- | ------------------- | --- | -------------------- | -------------- |
otherwise zero. The network is partitioned into multiple communities using some CDAs and
the community information is denoted by  = {C ∣1 ≤ i ≤ κ}, where κ is the total number of
i
communities in the network. The communities are considered as nonoverlapping, where two
communitiesC andC do not share any common nodes and can be written asC,C ∈  and
|     | i   | j   |     |     |     | i j |
| --- | --- | --- | --- | --- | --- | --- |
C ∩ C = ∅foralli,j ∈ {1,…,κ}.Thenodesmayhaveconnectionswithothernodesinsidethe
i j
community or outside the community. For a given communityC, a linke of nodeu ∈ C is
|     |     |     |     |     | i   | uv i |
| --- | --- | --- | --- | --- | --- | ---- |
called internal for u,v ∈ C and external for u ∈ C,v ∉ C. The internal and external con-
|     |     |     | i   |     | i i      |     |
| --- | --- | --- | --- | --- | -------- | --- |
|     |     |     |     |     |  and |     |
nectionsofthenodesforeach communityarestoredinthesets, i i,respectively.The
set of internal and external connections of a nodeu can be obtained through (u) and(u),
respectively. The total number of internal and external connections of the nodes in each
community is represented by I andO, respectively, whereas I(u) andO(u) provide the total
internal and external connections of u residing in a community. The degree of the nodes is
storedintheset andthedegree(u) ofthenodeu canbedenotedby(u)
= I(u) + O(u).
| The notations | are | provided | in Table | 2.  |     |     |
| ------------- | --- | -------- | -------- | --- | --- | --- |

|
5
KUMARIETAL.
| TABLE     | 2 Notationswiththeirdescriptions               |     |     |     |
| --------- | ---------------------------------------------- | --- | --- | --- |
| Notations | Descriptions                                   |     |     |     |
| G(,)    | Undirectednetwork                              |     |     |     |
| ,N       | Setofnodesandthetotalnumberofnodesinthenetwork |     |     |     |
| ,E       | Setoflinksandthetotalnumberoflinksinthenetwork |     |     |     |
| A         | AdjacencymatrixofsizeN×N                       |     |     |     |
| ,C i     | Setofcommunitiesandasinglecommunity            |     |     |     |
| C        | InternalconnectionsofthenodesinthecommunityC   |     |     |     |
| C        | ExternalconnectionsofthenodesinthecommunityC   |     |     |     |
(u),I(u) Setofinternalconnectionsofnodeu,thetotalnumberofinternalconnectionsofnodeu
(u),O(u) Setofexternalconnectionsofnodeu,thetotalnumberofexternalconnectionsofnodeu
| (u)       | Degreeofnodeu                          |            |     |     |
| ---------- | -------------------------------------- | ---------- | --- | --- |
| Ex(u,C)    | i Extractionscoreofnodeu∈C             |            | i   |     |
| ζ(u,C)     | Persistencescoreofnodeu∈C              |            |     |     |
| i          |                                        |            | i   |     |
| ζ(C)       | AveragepersistencescoreofthecommunityC |            |     |     |
| i          |                                        |            |     | i   |
| ζ(G)       | AveragepersistencescoreofthenetworkG   |            |     |     |
| coeff(u,C) | Internalclusteringcoefficientofnodeu∈C |            |     |     |
|            | i                                      |            |     | i   |
| σ(u,C)     | Safenessscoreofnodeu∈C                 |            |     |     |
| i          |                                        |            | i   |     |
| σ(C) i     | AveragesafenessscoreofthecommunityC    |            |     | i   |
| σ(G)       | AveragesafenessscoreofthenetworkG      |            |     |     |
| ρ          | Densityofacommunity                    |            |     |     |
|           | Hubdominance                           |            |     |     |
| 2.1 |      | Metrics                                | definition |     |     |
Therearesomemetricstodescribeandanalyzedeceptionapproaches.Thedescriptionofsome
| of the metrics | is provided  | in the following | subsections: |     |
| -------------- | ------------ | ---------------- | ------------ | --- |
| 2.1.1          | | Extraction | score            |              |     |
Extractionscoreisanode‐basedmetricwhichisusedtofindthemaximumnumberofexternal
connections of a node from a single outside community. For a node u in community C, the
i
extractionscoreEx(u,C) i isthemaximumconnectionwithanyoutsidecommunityC.Tofind j
Ex(u,C), it is needed to find neighboring community nodes attached with node u. All the
i
| neighboring | communities | idC is stored | in a set | ω.  |
| ----------- | ----------- | ------------- | -------- | --- |
j
}.
|     | ω   | = {C: j Cis neighboring community of node u,C j |     | j ∈ |
| --- | --- | ----------------------------------------------- | --- | --- |
Thereafter,thetotalnumberofexternallinksofuiscalculatedcommunitywise(storedinι)and
the maximum score returns Ex(u,C). The extraction score can be formulated as
i

|
6
KUMARIETAL.
∑A
|     |     |     | ι=  | ,∀ C ∈ ω, |     |
| --- | --- | --- | --- | --------- | --- |
u,v j
v∈Cj (1)
Ex(u,C)=maxι.
i
C
i
| 2.1.2 | | Persistence | score |     |     |     |
| ----- | ------------- | ----- | --- | --- | --- |
Persistenceisanothernode‐basedmetricthatisusedtofindtheintensityofbelongingnessofa
nodetoitscommunity.Itdependsonthreefactors:extractionscoreofthenode,strengthofits
internal connectivity, and clustering coefficient of the node within the community (by con-
sidering internal nodes and links). After the consideration of all the factors, persistence score
ζ(u,C) i of each nodeu residing in communityC can be formulated as
i
⎛ ⎞
|     |     |          | 1 I(u) | 1                 |     |
| --- | --- | -------- | ------ | ----------------- | --- |
|     |     | ζ(u,C) = | ⎜      | × + coeff (u,C)⎟, | (2) |
i 2⎝Ex(u,C) i ⎠
|     |     |     | i   | K(u) |     |
| --- | --- | --- | --- | ---- | --- |
whereI(u)istheinternalconnectionsofuinC,Ex(u,C)theextractionscoreofu ∈ C,K(u)the
|     |     |     |     | i i | i   |
| --- | --- | --- | --- | --- | --- |
degree of u in the network, and coeff (u,C) the internal clustering coefficient of u ∈ C. The
|     |     |     | i   |     | i   |
| --- | --- | --- | --- | --- | --- |
internalclusteringcoefficientofthenodesinC isstoredinthesetcoeffC i.Theζ(u,C ) decreases
|                                                                                       |     |     |     | i   | i   |
| ------------------------------------------------------------------------------------- | --- | --- | --- | --- | --- |
| withtheincreasingvalueofextractionscoreEx(u,C),whereasitisincreasingwiththeincreasein |     |     |     | i   |     |
thevalueofI(u)ofthefirstterminEquation(1).Hence,theinternalconnections,I(u),foranode
u shouldbemorethanitsextractionscore,Ex(u,C).Thesecondtermisusedfornormalizingthe
i
firstterm.Theinternalclusteringcoefficientofu,coeff (u,C),alsoincreasesthepersistencescore
i
ofnodeu.Ifanodeishighlyclusteredwiththeinternalnodesinsidethecommunity,thenitwill
increasetheintensityofbelongingnessofthenodeinsidethecommunity.Thepersistencescoreisa
node metric which considers the strength of the node inside the community; hence, it is out-
performed other metrics, such as modularity, conductance, Cut Ratio, and so forth. The network
persistence score is calculated by summing the persistence score of all the nodes.
| 2.1.3 | | Network | persistence | boundary | conditions |     |
| ----- | --------- | ----------- | -------- | ---------- | --- |
• CaseI:Foranisolatednodeu,thepersistencescoreζ(u,C)iszero.Ifanodeisisolatedfromthe
i
network,thenthevaluesofallthecomponents,I(u),(u),Ex(u,C),andcoeff (u,C),become
i i
zero which leads to a zero persistence score. Hence, the lower bound of persistence score is 0.
• CaseII:Forthenodeuthatdoesnothaveexternalconnections,thepersistencescoreζ(u,C)
i
1coeff
is equal to the internal clustering coefficient coeff (u,C) (i.e., ζ(u,C) i = (u,C)). i
|     |     |     |     | i 2 |     |
| --- | --- | --- | --- | --- | --- |
• CaseIII:Whenanodeu isaninternalnodeandalsoapartofaclique,thenthescoreofallthe
parameters, I(u),(u),Ex(u,C), and coeff (u,C), is equal to 1. By putting all the values in
|     |     |     | i   | i   |     |
| --- | --- | --- | --- | --- | --- |
1(1
Equation (1), persistence score ζ(u,C) i = + 1) = 1. Therefore, the upper bound of persis-
2
| tence | score is 1. |       |     |     |     |
| ----- | ----------- | ----- | --- | --- | --- |
| 2.1.4 | | Safeness  | score |     |     |     |
anothernode‐basedmetricwhich
| Safenessis |     |     |     | is usedtomeasurethedeceptionor | hidingscore |
| ---------- | --- | --- | --- | ------------------------------ | ----------- |
ofanodeinsideacommunity.Weprovideascoreσ(u,C)toeachnodeuinthecommunityC.
i i

|
KUMARIETAL. 7
Ifanodehasa largersafenessscore,thenthat nodewillnotbe easilyidentifiable fromCDAs,
andweconsiderthat nodeasasafenodeinside thecommunityC.Safenessofanodeσ(u,C)
i
in a community C is computed by considering two factors: a total number of indirect con-
i
nectionsofthenodeuwithothernodesinsidethecommunityC andatotalnumberofexternal
i
connections of the nodeu with other communities. If we can increase any of the factors, then
we will be able to increase the safeness score of the nodeu inside the communityC.39
i
1
R
u
Ci − I(u)
1 O(u)
σ(u,C) = + , (3)
i
2 ∣C ∣ − 1 2 (u)
i
where∣R
u
Ci∣isthenumberofnodesofC
i
reachablefromupassingonlythroughnodesinsideC
i
,
I(i) the number of internal links of node u ∈ C,O(u) the number of external links of node
i
u ∈ C, and (u) the degree of nodeu.
i
∣RCi∣ − I(u) computes thedifferencebetween thenumberofnodesreachablefromnodeu
u
toalltheothernodesinsidethecommunityC andtotalinternallinksofnodeu ∈ C.Thevalue
i i
of the parameter∣R
u
Ci∣ − I(u) ranges between[0,∣C
i
∣ − 1]. The first factor ensures that in an
idealcase,thenodeuisabletoreachallothernodesinthecommunityC.Thesecondfactoris
i
usedtoconfusetheCDAs.Itensuresthatu isdiverseintermsofitsconnections,thatis,there
are a large number of external links fromu to other communities that are leading to a better
community hiding. The community safeness σ(C) and the network safeness σ(G) are calcu-
i
lated by summing the safeness score of all the nodes inside the community and the network,
respectively.
2.1.5 | Safeness score boundary condition
• Case I: The worst‐case scenario is when u is unable to reach any other nodes inside com-
munity C or outside the community, and it results in score 0. Hence, the lower bound of
i
safeness score is 0.
• Case II: The best‐case scenario is when u is reachable to all the other nodes via indirect
connection by following the path containing internal links in community C (i.e.,
i
∣R
u
Ci∣ − I(u) = ∣C
i
∣ − 1).Apartfromthat,alltheconnectionsofnodeuaremadethroughan
externallink(i.e.,O(u) = (u)).Bymaintainingboththecriteria,safenessscorebecomes1,
and it is the upper bound of safeness score, that is, completely safe.
2.2 | Running example
In Figure 1, the community information of a Domestic terror network is unveiled using Lou-
vain CDA. The nodes are shown through different colors based on their presence in the
communities. The communities, C,C ,C , and C , are shown through purple, blue, orange,
1 2 3 4
andgreencolor,respectively.Thepersistencescoreζ(u,C)ofeachnodeineachcommunityC
i i
is computed using Equation (1). The persistence score, ζ(C), of each community C is com-
i i
puted by taking an average of the persistence score of all the nodes in the community C.
i
Similarly,persistencescoreζ(G)ofthenetworkG iscomputedbytakingtheaverageofallthe
persistence scores of each communityC. The safeness score σ(G) of the networkG is calcu-
i
lated in the same manner as persistence score ζ(G) of the networkG.

|
8 KUMARIETAL.
FIGURE 1 Domesticterrornetworkconsistsof32nodesandispartitionedintofourcommunitiesusingthe
Louvaincommunitydetectionalgorithm.Thenodesinthedifferentcommunitiesareshownthroughvarious
colors(orange,blue,green,andpurple).Thepersistenceandsafenessscoresofthenodes,communities,and
networkarecomputedfortheDomesticterrornetwork[Colorfigurecanbeviewedatwileyonlinelibrary.com]
The persistence scoreζ(u,C) and the safeness scoreσ(u,C) assess the level of deception of
i i
thenodeu ∈ C againstsomeCDAs.Higherthevalueofζ(u,C) ofnodeu,lowerthedeception
i i
score of nodeu ∈ C against CDAs. In contrast, the higher the value of σ(u,C) of nodeu, the
i i
higherthedeceptionscoreϕ(u) ofnodeu ∈ C againstCDAs.Theprimeconcernofthispaperis
i
todevisealgorithmssuchthatitimprovesthedeceptionscoreofthenodesinsidethecommunity.
At this very moment, the question comes up about how to address the community de-
ception. There may be multiple approaches to solve the problem. One way to approach the
problem would be to maximize the deception score, ϕ. Another approach may be based on
some rewiring technique with the nodes inside or outside the community. In this paper, we
have considered both methods. The suggested research of this paper would be conducted as
follows. First, we will find the community structure of the network using multiple CDAs and
willdestabilizethecommunitystructureofthesystem.Theoptimizationproblemforcommunity
destabilization is based on the maximization of persistence loss (MPL) and maximization of
safeness gain (MSG). Second, we will use some smart rewiring techniques to hide the com-
munity of the nodes in the network. The deception techniques are explained in detail in the
following sections.
2.3 | Safeness‐based deception
Safeness score is used to measure the deception of a node in the community from CDAs. A
more significant value of safeness score indicates better deception of the node and considered
as safe inside the community. The application of safeness‐based deception is the deception of
security personnel in terror network, deception of some agents in the social network to detect
fraudulent activity, deception of nodes in a neural network for better analysis, and so on. The
safeness score of the node is evaluated using the steps in Section 2.1.4. The rewiring strategy
enhancesthesafenessscoreofthenodesinthecommunityandfurthermaximizesthesafeness
score of the network.39

|
9
KUMARIETAL.
| 2.3.1 | | Maximization |     |     | of safeness | gain |
| ----- | -------------- | --- | --- | ----------- | ---- |
σ(CMSG)
A deception function, Φ, is defined as the safeness gain, − σ(C), where C is the
original community and CMSG the community updated after the rewiring operation. So, the
algorithm works greedily, findinganedgeupdate atevery step which accounts for thehighest
valueforsafenessgain.Intherewiringoperation,oneexistingconnectionisremoved,andone
new connection is established between two nodes in the network. The removal of the existing
connectionisdonebasedontheapproachprovidedinReference[39].Whiletheadditionofthe
new connection is provided using the concept of extraction score explained in Section 2.1.1.
There may be four aspects for rewiring operation: intracommunity edge addition, inter-
community edge addition, intercommunity edge deletion, and intracommunity edge deletion.
Intercommunity edge addition always causes a gain in safeness score, and intercommunity
edge deletion will decrease the safeness score. Intracommunity edge addition or edge deletion
| may increase | or  | lower | the safeness | score. |     |
| ------------ | --- | ----- | ------------ | ------ | --- |
Anodeuispicked,andthefractionofexternalconnectioniscalculated.Afterthat,anodev
with maximum extraction score Ex(v,C) j from the neighboring communityC j is chosen for a
new connection between nodesu and v. Now, an intracommunity link e m,n between nodesm
andn with maximum O(m,C) + O(n,C) score is removed from the network.39 The
|          |           |             | 2(m)((m)−1)     |         | 2(n)((n)−1)    |
| -------- | --------- | ----------- | ----------------- | ------- | ---------------- |
| rewiring | operation | is repeated |                   | a large | number of times. |
| 2.4 |    | Proposed  |             | persistence‐based |         | deception        |
Persistence is a measure to compute the chances for a vertex to be in its community. If the
communityofagroupofnodesalwayslietogetheragainstmultipleCDAs,thenthosenodesare
called a CC. The persistence score of the nodes evaluated using the formula in Section 2.1.2.
MoremassivepersistencescoresdemonstratestrongerCCandeasilyidentifiablebyCDAs.We
aimtohidethenodesfromCDAs;hence,theproposedstrategyminimizesthepersistencescore
| and maximizes |                | the persistence |     | loss of        | the graph. |
| ------------- | -------------- | --------------- | --- | -------------- | ---------- |
| 2.4.1         | | Maximization |                 |     | of persistence | loss       |
The deception function ϕ is defined as the persistence loss, ζ(C) − ζ(C)MPL, whereC is the
i i i
originalcommunityandCMPLtheupdatedcommunityafterapplyingthecommunitydeception
i
approach. The approach works greedily, moving a node from one community to another
community with a maximum extraction score (explained in Section 2.1.1). The above steps
| result in | maximum | persistence |     | loss. |     |
| --------- | ------- | ----------- | --- | ----- | --- |
MPLwilldestabilizethecommunitystructureofthenetwork.Wearetryingtomaximize
the loss in persistence after updating the network so that we can disrupt the original
community structure as much as possible. The node is shifted from one community to
another community if the persistence score of the node is getting decreased. The process is
repeated for all the nodes in the network. The detailed description is also provided in
| Algorithm | 1.  |     |     |     |     |
| --------- | --- | --- | --- | --- | --- |

|
10
KUMARIETAL.
Algorithm1Maximizationofpersistenceloss
1:Input:G,,,total←0,count←0,maxIt←10,MPL←[],MPL←[]
2:Output:ζMPL(G),MPL,MPL
| 3:whileitern<maxIt | do  |     |
| ------------------ | --- | --- |
4: count←count+1
5: total←0
6: MPL←
| 7: foru | in do |     |
| ------- | ------ | --- |
8: comm←Cu
i
| 9: Calculatetheζ(u,C) | i andneighborsNe(u) | ofu |
| --------------------- | ------------------- | --- |
10 nei_com←[]
| 11: ifζ(u,C)==1andlen(Ne(u))==0 | i   | then |
| ------------------------------- | --- | ---- |
12: total←total+ζ(u,C)
i
13: continue
14: endif
15: S ←0
1
| 16: forv | inNe(u) do |     |
| -------- | ---------- | --- |
| 17: S    | =S +ζ(v,C) |     |
| 1        | 1 i        |     |
nei_com.append(Cv)
| 18: | i   |     |
| --- | --- | --- |
19: endfor
20: fornewinnei_com do
21: Calculatetheζ(u,new)
| 22: FindnewneighborsNe′(u) | ofu |     |
| -------------------------- | --- | --- |
23: S ←0
2
| 24: forv | inNe′(u) do |     |
| -------- | ----------- | --- |
| 25: S    | =S +ζ(v,C)  |     |
|          | 2 2 i       |     |
26: endfor
| 27: ifζ(u,C)>ζ(u,new)              | andS >S | then              |
| ---------------------------------- | ------- | ----------------- |
|                                    | i 1     | 2                 |
| 28: Assignζ(u,C)←ζ(u,new),comm←new | i       | andMPL.append(u) |
29: else
30: Cu←comm
i
31: endif
32: total←total+ζ(u,C)
i
33: C′.append(comm)andMPL.append(C′)
|     | i   | i   |
| --- | --- | --- |
34: endfor
35: endfor
36:endwhile
| 37:ζMPL(G)← | total |     |
| ----------- | ----- | --- |
N
returnζMPL(G),MPL,MPL
| 2.4.2 | | Custom rewiring |     |
| ------- | --------------- | --- |
Inthecustomrewiringapproach,werewirethelinksofsomeselectednodes.Intherewiring
operation, one node is selected, and one of its connections is removed and added to some
other node in the network. In the MPL strategy, some nodes are shifted to the neighboring

|
11
KUMARIETAL.
communitytomaximizethepermanenceloss.AftertheMPLapproach,thenodesareshifted
to a neighboring community, but the structure of the network is not changing. After the
custom rewiring approach, the structure of the network is altered. In this approach, the
connections of all the shifted nodes are rewired to maximize the permanence loss and
| maximize the safeness | gain. |     |
| --------------------- | ----- | --- |
The detailed description of the steps involved in the custom rewiring approach is pro-
vided in Algorithm 2. For the rewiring operation, one node is chosen from the old com-
munity, and another is from the new community. Among all the shifted nodes, a nodeu is
picked and its neighbors from the old community, as well as the new community, are ob-
tained. A node v with maximum I(v) is chosen for link removal operation. And, a node w
with minimuminternal connection ispicked for anewconnection after rewiring operation.
The rewiring operation is repeated for all the shifted nodes in the neighboring community
| after the MPL strategy. |     |     |
| ----------------------- | --- | --- |
Algorithm2Customrewiringapproach
1:Input:G,MPL,,MPL
2:Output:RewirednetworkGMPL
| 3:foru inMPL do       |                           |     |
| ---------------------- | ------------------------- | --- |
| 4: FindneighborsNe(u)  | ofnodeu                   |     |
| 5: Findtheneighborsofu | intheoldcommunityNe(u)old |     |
6: FindtheneighborsNe(u)MPL ofu afterupdationthroughMPLstrategy
7: iflen(Ne(u)old)>1andlen(Ne(u)MPL)>1then
| 8: value_d←0 andvalue_a←1000 |     |     |
| ---------------------------- | --- | --- |
| 9: forv inNe(u)old           | do  |     |
(v) ofnodev
10: Findinternalconnections
| 11: ifvalue_d≤I(v) | and(v)>1andu≠v | then |
| ------------------ | --------------- | ---- |
12: value_d←I(v)
13: del_node←v
14: endif
15: endfor
| 16: forw inNe(u)MPL         | do           |     |
| --------------------------- | ------------ | --- |
| 17: Findinternalconnections | (w) ofnodew |     |
| 18: ifvalue_a≤I(w)          | then         |     |
19: value_a←I(w)
20: add_node←w
21: endif
22: endfor
| 23: Removeedgee | _   |     |
| --------------- | --- | --- |
u,del node
| 24: Addedgee | _   |     |
| ------------ | --- | --- |
u,add node
25: endif
26:endfor
returnGMPL

|
12
KUMARIETAL.
| 3 | | EXPERIMENTAL |     | RESULTS |     |     |
| --- | ------------ | --- | ------- | --- | --- |
Weconsidertwoundirectedandunweightedreal‐worldnetworkstakenfromdiversedomains.
The data sets are represented by two tuples (N,E): Domestic terror network40 (32,89) and
Dolphins41(62,159).Wechooseseven popularCDAs,Greedy modularity,28labelpropagation,
| LPA,29 | K_clique,2 | Bisection,30 | Girvan–Newman,7 | Louvain,8 |                     |
| ------ | ---------- | ------------ | --------------- | --------- | ------------------- |
| Asyn   |            |              |                 | and       | to detect community |
structurefromeachnetwork.Inthepaper,twoparameters,persistenceandsafenessscores,are
used to measure the community deception of the network. Two objective functions are pro-
posed, maximization in persistence loss and maximization in safeness gain, to reduce the
numberofCCs.Thenormalizedmutualinformation(NMI)scoreisusedtomeasuretheextent
| of possible | deception | using the proposed | approaches. |     |     |
| ----------- | --------- | ------------------ | ----------- | --- | --- |
real‐world
| 3.1 | | Community | structure | of the | data | sets |
| ----- | --------- | --------- | ------ | ---- | ---- |
The community of all the discussed data sets is obtained through seven CDAs. When the
network topology is changing due to rewiring operations, then the community information of
| nodes is | also changing | accordingly. |     |     |     |
| -------- | ------------- | ------------ | --- | --- | --- |
| 3.1.1    | | Dolphins    | data set     |     |     |     |
ThecommunitystructureoftheDolphinsnetworkisunveiledthroughdifferentCDAsinFigure2.
Theminimumandthemaximumnumberofcommunitiesis2and8thatisobtainedbyBisection
approach and Asyn LPA approach, respectively. The total number of communities  obtained
through the custom rewiring strategies is smaller than or equal to the communities obtained
throughotherstrategiesforalltheCDAs.IntheMPLstrategy,weareonlyshiftingnodesfromone
community to the neighboring community without alteration in the network structure; hence,
scoreisthesameforMPLandinitialnetworkcase.TheMSGstrategyincreasesthevalueof for
Asyn LPA (8, others have value 6) and K_clique (4, others have value 3) CDAs.
| 3.1.2 | | Domestic | terror network | data set |     |     |
| ----- | ---------- | -------------- | -------- | --- | --- |
The community structure of the Domestic terror network is unveiled through different CDAs in
Figure3.Theminimumandthemaximumnumberofcommunitiesis2and7thatisobtainedby
Bisection approach and Asyn LPA approach, respectively. A total number of communities 
obtainedthroughtheMSGstrategyandcustomrewiringstrategiesaresmallerthanorequaltothe
communities obtained through other strategies for all the CDAs except K_clique. In the MPL
strategy,weareonlyshiftingnodesfromonecommunitytotheneighboringcommunitywithout
alterationinthenetworkstructure;hence,
scoreisthesameforMPLandinitialnetworkcase.
Safeness‐based
| 3.2 | |     | results |     |     |     |
| ----- | --- | ------- | --- | --- | --- |
G
The safeness score σ(G) of the graph is evaluated by using the formula provided in
Section 2.1.4. The value ofσ(G) is evaluated for the network obtained after MPL strategy, the

|
13
KUMARIETAL.
| (A) | (B) | (C)  | (D)   |
| --- | --- | ---- | ----- |
| (E) | (F) | (G)  | (H)   |
| (I) | (J) | (K)  | (L)   |
| (M) | (N) | (O)  | (P)   |
| (Q) | (R) | (S)  | (T)   |
| (U) | (V) | (W)  | (X)   |
| (Y) | (Z) | (Z’) | (Z’’) |
(Continued)
FIGURE 2

|
14 KUMARIETAL.
network obtained after rewiring operation through MSG strategy and the network rewired
throughcustomrewiringstrategy.Thegoalofeachstrategyistoenhancethesafenessscoreof
the graph. If a node u has higher σ(u,C), then it will not be identifiable from CDAs. The
i
percentagegaininσ(G) iscalculatedforallthestrategiesforthenetworkpartitionedthrough
various CDAs.
3.2.1 | Dolphins data set
The initial value of safeness score σ(G) of the Dolphins network is compared with all the
discussed strategies, and safeness gain in σ(G) is obtained for all the CDAs (Table 3). In the
MPLstrategy,theminimumandthemaximumvalueofsafenessgainisapproximately27%and
47%forLouvainandBisectionCDAs,respectively.Theaveragesafenessgainisapproximately
38%forMPLstrategy.Inthecustomrewiringstrategy,theminimumandthemaximumvalue
of safeness gain is approximately 0.5% and 11% for Girvan–Newman and Asyn LPA CDAs,
respectively.Theaveragesafenessgainisapproximately7%oftheinitialvalueofσ(G).Forthe
MSG strategy, the minimum and the maximum value of safeness score is approximately 0.5%
and 5% for Girvan–Newman and Asyn LPA CDAs, respectively. The average safeness gain is
approximately 2.5% for MSG strategy than that of the initial value σ(G).
3.2.2 | Domestic terror network data set
The initial value of safeness score σ(G) of the Domestic Terror network is compared with all
thediscussedstrategies,andsafenessgaininσ(G)isobtainedforalltheCDAs(Table4).Inthe
MPLstrategy,theminimumandthemaximumvalueofsafenessgainisapproximately5%and
32% for Bisection and Asyn LPA CDAs, respectively. The average safeness gain is approxi-
mately 19% for MPL strategy. For the custom rewiring strategy, the minimum and the max-
imum value of safeness gain is approximately 3% and 19% for Bisection and Asyn LPA CDAs,
respectively. For the custom rewiring strategy, the average safeness gain is approximately 11%
of the initial σ(G). For the MSG strategy, the minimum and the maximum value of safeness
score is approximately 0.11% and 2.6% for Girvan–Newman and Bisection CDAs, respectively.
Theaveragesafenessgainisapproximately1.27%forMSGstrategythanthatoftheinitialvalue
σ(G). The safeness gain for all the discussed data sets under different strategies is analyzed.
From the results, we can infer that the MPL strategy is performing best, and MSG strategy is
performing worst in terms of deception of nodes against CDA in the network.
FIGURE 2 CommunitystructureoftheDolphinsnetworkdetectedthroughsevenCDAsforfourdifferent
cases:initialnetwork,communityinformationobtainedafterMPLstrategy,thenetworkobtainedaftercustom
rewiringapproach,andrewirednetworkafterMSGstrategy.(A–D)Thecommunitystructureofthenetwork
obtainedthroughGreedyapproachforallthefourcasesisplotted.Similarly,(E–H),(I–L),(M–P),(Q–T),(U–X),
and(Y,Z,Zʹ,Z")demonstratethecommunityofthenetworkobtainedthroughGirvan–Newman,label
propagation,AsynLPA,Bisection,K_clique,andLouvainCDAs,respectively,forallthediscussedcases.CDA,
communitydetectionalgorithm;AsynLPA,asynchronouslabelpropagationalgorithm;MPL,maximizationof
persistenceloss;MSG,maximizationofsafenessgain[Colorfigurecanbeviewedatwileyonlinelibrary.com]

|
15
KUMARIETAL.
| (A) | (B) | (C)  | (D)   |
| --- | --- | ---- | ----- |
| (E) | (F) | (G)  | (H)   |
| (I) | (J) | (K)  | (L)   |
| (M) | (N) | (O)  | (P)   |
| (Q) | (R) | (S)  | (T)   |
| (U) | (V) | (W)  | (X)   |
| (Y) | (Z) | (Z’) | (Z’’) |
FIGURE 3 (Continued)

|
16
KUMARIETAL.
| TABLE            | 3 SafenessgainforDolphinsnetwork |                |        |                |        |
| ---------------- | -------------------------------- | -------------- | ------ | -------------- | ------ |
|                  |                                  | σ(G)           | %Gain  | %Gain          | %Gain  |
| CDA              |                                  | Initialnetwork | MPL    | customrewiring | MSG    |
| Greedymodularity |                                  | 0.4445         | 28.055 | 7.894          | 1.458  |
| Labelpropagation |                                  | 0.4334         | 32.766 | 8.787          | 1.250  |
| Girvan–Newman    |                                  | 0.4366         | 42.213 | 0.5167         | 0.5107 |
| AsynLPA          |                                  | 0.3996         | 47.236 | 11.429         | 5.632  |
| Bisection        |                                  | 0.4070         | 47.285 | 11.202         | 2.481  |
| K_clique         |                                  | 0.3808         | 43.565 | 7.682          | 4.356  |
| Louvain          |                                  | 0.4440         | 27.603 | 3.805          | 1.878  |
Abbreviations:CDA,communitydetectionalgorithm;AsynLPA,asynchronouslabelpropagationalgorithm;MPL,
maximizationofpersistenceloss;MSG,maximizationofsafenessgain.
Persistence‐based
3.3 | results
We calculated the persistence score of each node inside the community and found that the
moremassivepersistencescoreleadstoamoresignificantnumberofCCsinthenetwork.Ifwe
wanttohidethecommunityofthenodesinthenetwork,thenweneedtodestabilizetheCCs.
AnobjectivefunctionisproposedinSection2.4.1tomaximizethepersistencelossofthenodes
and the network. We calculated the information corresponding to each community (C) parti-
tioned through different CDAs before and after the application of all the strategies: MPL,
| custom | rewiring, and | MSG.     |     |     |     |
| ------ | ------------- | -------- | --- | --- | --- |
| 3.3.1  | | Dolphins    | data set |     |     |     |
The initial value of persistence score ζ(G) of the Dolphins network is compared with all the
discussed strategies, and percentageloss inζ(G) is obtained for all the CDAs (Table 5). In the
MPL strategy, theminimum and the maximum value of persistence loss is approximately 61%
and 69% for Greedy modularity and label propagation CDAs, respectively. The average per-
sistence loss is approximately 65% for MPL strategy. For the custom rewiring strategy, the
minimum and the maximum value of persistence loss is approximately 5% and 14% for
Girvan–Newman
and Asyn LPA CDAs, respectively. The average persistence loss is
FIGURE 3 CommunitystructureoftheDomesticterrornetworkdetectedthroughsevenCDAsforfour
differentcases:initialnetwork,communityinformationobtainedafterMPLstrategy,thenetworkobtainedafter
customrewiringapproach,andrewirednetworkafterMSGstrategy.(A–D)Thecommunitystructureofthe
networkobtainedthroughGreedyapproachforallthefourcasesisplotted.Similarly,(E–H),(I–L),(M–P),
(Q–T),(U–X),and(Y,Z,Zʹ,Z")demonstratethecommunityofthenetworkobtainedthroughGirvan–Newman,
labelpropagation,AsynLPA,Bisection,K_clique,andLouvainCDAs,respectively,forallthediscussed
cases.CDA,communitydetectionalgorithm;AsynLPA,asynchronouslabelpropagationalgorithm;MPL,
maximizationofpersistenceloss;MSG,maximizationofsafenessgain[Colorfigurecanbeviewedat
wileyonlinelibrary.com]

|
17
KUMARIETAL.
| TABLE            | 4 SafenessgainforDomesticterrornetwork |                |        |                |        |
| ---------------- | -------------------------------------- | -------------- | ------ | -------------- | ------ |
|                  |                                        | σ(G)           | %Gain  | %Gain          | %Gain  |
| CDA              |                                        | Initialnetwork | MPL    | customrewiring | MSG    |
| Greedymodularity |                                        | 0.3617         | 23.284 | 10.582         | 0.5841 |
| Labelpropagation |                                        | 0.3669         | 27.424 | 16.830         | 0.5719 |
| Girvan–Newman    |                                        | 0.3790         | 10.535 | 12.152         | 0.1182 |
| AsynLPA          |                                        | 0.3350         | 32.554 | 19.752         | 2.456  |
| Bisection        |                                        | 0.4331         | 4.931  | 3.484          | 2.675  |
| K_clique         |                                        | 0.3860         | 14.156 | 4.927          | 2.026  |
| Louvain          |                                        | 0.3617         | 23.284 | 10.479         | 0.4599 |
Abbreviations:CDA,communitydetectionalgorithm;AsynLPA,asynchronouslabelpropagationalgorithm;MPL,
maximizationofpersistenceloss;MSG,maximizationofsafenessgain.
approximately 9.27% of the initial value ofζ(G). For the MSG strategy, the minimum and the
maximum value of persistence loss is approximately 0.111% and 4.6% for Greedy modularity
and Bisection CDAs, respectively. The average persistence loss is approximately 2.7% for MSG
| strategy | than that of | the initial value ζ(G). |          |     |     |
| -------- | ------------ | ----------------------- | -------- | --- | --- |
| 3.3.2    | | Domestic   | terror networks         | data set |     |     |
Theinitialvalueofpersistencescoreζ(G)oftheDomesticterrornetworkiscomparedwithall
the discussed strategies, and percentage loss in ζ(G) is obtained for all the CDAs (Table 6).
IntheMPLstrategy,theminimumandthemaximumvalueofpersistencelossisapproximately
Girvan–Newman
71% and 79% for Bisection and CDAs, respectively. The average persistence
loss is approximately 76% for MPL strategy. For the custom rewiring strategy, the minimum
and the maximum value of persistence loss is approximately 12.75% and 32% for label
TABLE 5 ComparisoninpersistencescorebeforeandaftertheedgeupdateoperationforDolphinsnetwork
undervariousstrategies:MPL,customrewiring,andMSG
|                  |     | ζ(G)           | %Loss  | %Loss          | %Loss |
| ---------------- | --- | -------------- | ------ | -------------- | ----- |
| CDA              |     | Initialnetwork | MPL    | customrewiring | MSG   |
| Greedymodularity |     | 0.1371         | 61.222 | 8.425          | 0.111 |
| Labelpropagation |     | 0.1250         | 68.752 | 7.164          | 3.370 |
Girvan–Newman
|           |     | 0.1925 | 66.960 | 4.995  | 0.4470 |
| --------- | --- | ------ | ------ | ------ | ------ |
| AsynLPA   |     | 0.1672 | 68.235 | 13.661 | 2.582  |
| Bisection |     | 0.0834 | 63.378 | 9.271  | 4.584  |
| K_clique  |     | 0.0647 | 63.522 | 12.257 | 1.380  |
| Louvain   |     | 0.1341 | 66.943 | 9.133  | 6.433  |
Abbreviations:CDA,communitydetectionalgorithm;AsynLPA,asynchronouslabelpropagationalgorithm;MPL,
maximizationofpersistenceloss;MSG,maximizationofsafenessgain.

|
18
KUMARIETAL.
TABLE 6 ComparisoninpersistencescorebeforeandaftertheedgeupdateoperationforDomesticterror
networkundervariousstrategies:MPL,customrewiring,andMSG
|                  | ζ(G)           | %Loss  | %Loss          | %Loss  |
| ---------------- | -------------- | ------ | -------------- | ------ |
| CDA              | Initialnetwork | MPL    | customrewiring | MSG    |
| Greedymodularity | 0.2628         | 77.567 | 21.924         | 5.869  |
| Labelpropagation | 0.2368         | 73.515 | 12.747         | 0.3648 |
| Girvan–Newman    | 0.2772         | 78.824 | 16.460         | 1.311  |
| AsynLPA          | 0.2114         | 74.856 | 30.591         | 6.865  |
| Bisection        | 0.2056         | 71.351 | 13.029         | 4.876  |
| K_clique         | 0.2111         | 76.761 | 32.197         | 3.427  |
| Louvain          | 0.2628         | 77.567 | 16.091         | 11.835 |
Abbreviations:CDA,communitydetectionalgorithm;AsynLPA,asynchronouslabelpropagationalgorithm;MPL,
maximizationofpersistenceloss;MSG,maximizationofsafenessgain.
propagation and K_clique CDAs, respectively. The average persistence loss is approximately
20%forcustomrewiringstrategythanthatoftheinitialvalueζ(G).FortheMSGstrategy,the
minimumandthemaximumvalueofpersistencelossisapproximately0.36%and12%forlabel
propagationandLouvainCDAs,respectively.Theaveragepersistencelossisapproximately5%
| of the initial value ofζ(G). |     |     |     |     |
| ---------------------------- | --- | --- | --- | --- |
The persistence loss for all the discussed data sets under different strategies is
analyzed. From the results, we can infer that the MPL strategy is performing best, and
MSG strategy is performing worst in terms of reduction in the CCs and maximization in
persistence loss.
| 3.4 | NMI score |     |     |     |     |
| --------------- | --- | --- | --- | --- |
Each CDA partitions the network into multiple communities. Some CDAs divide the net-
work into a large number of communities while some are dividing the network into a
smaller number of communities. We need to know the real ground truth of the commu-
nitiestofindactualcommunityinformationofthenodesinthenetwork.Afterthat,wefind
mutual community information (MI) shared between the true community label and pre-
dicted community label. Higher the value of MI corresponds to correct community in-
formation detected through the CDA. In this paper, we want to deceive the CDA, so lower
the value of MI is preferred. The normalization of the MI is called Normalized Mutual
Information (NMI) and the value of 0 ≤ NMI ≤ 1. The score NMI = 0 corresponds to no
mutualinformationand NMI = 1correspondstoperfectcorrelation.Inthispaper,wehave
usedthemethodofNMIcalculationprovidedinReference[42].Thismethodworksfortwo
independent community labeled nodes of the same data set that is provided as an input
without the knowledge of ground truth community information. In this function, mutual
information is normalized by some generalized mean of the true labels and predicted label
definedbytheaveragemethod.ThepermutationofthelabelvaluesdoesnotaffecttheNMI
score.

|
19
KUMARIETAL.
| TABLE            | 7 NMIscoreforDolphinsnetwork |        |     |                |        |
| ---------------- | ---------------------------- | ------ | --- | -------------- | ------ |
| CDA              |                              | MPL    |     | Customrewiring | MSG    |
| Greedymodularity |                              | 0.5202 |     | 0.6718         | 0.8082 |
| Labelpropagation |                              | 0.5249 |     | 0.4763         | 0.7484 |
| Girvan–Newman    |                              | 0.1138 |     | 0.6372         | 0.7081 |
| AsynLPA          |                              | 0.6053 |     | 0.5128         | 0.7085 |
| Bisection        |                              | 0.1308 |     | 0.5413         | 0.3175 |
| K_clique         |                              | 0.3901 |     | 0.6687         | 0.8234 |
| Louvain          |                              | 0.4426 |     | 0.6069         | 0.9215 |
Abbreviations:CDA,communitydetectionalgorithm;AsynLPA,asynchronouslabelpropagationalgorithm;MPL,
maximizationofpersistenceloss;MSG,maximizationofsafenessgain;NMI,NormalizedMutualInformation.
| 3.4.1 | | Dolphins | data set |     |     |     |
| ----- | ---------- | -------- | --- | --- | --- |
The NMI score of the community detected through a CDA for the initial network and the
communityinformationofthenodesaftereachstrategyiscalculatedindividually(Table7).In
theMPLstrategy,theminimumandthemaximumvalueofNMIscoreis0.1138and0.6053for
Girvan–NewmanandAsynLPACDAs,respectively.TheaverageNMIscoreis0.3897forMPL
strategy.Forthecustomrewiringstrategy,theminimumandthemaximumvalueofNMIscore
is 0.4763 and 0.6718 for label propagation and Greedy modularity CDAs, respectively. The
average score of NMI is approximately 0.5878 for custom rewiring strategy. For the MSG
strategy,theminimumandthemaximumvalueofNMIscoreis0.3175and0.9215forBisection
and Louvain CDAs, respectively. The average score of NMI is 0.7193 for MSG strategy.
| 3.4.2 | | Domestic | terror network | data set |     |     |
| ----- | ---------- | -------------- | -------- | --- | --- |
The NMI score of the community detected through a CDA for the initial network and the
community information of the nodes after each strategy is calculated individually (Table 8).
| TABLE            | 8 NMIscoreforDomesticterrornetwork |        |     |                |        |
| ---------------- | ---------------------------------- | ------ | --- | -------------- | ------ |
| CDA              |                                    | MPL    |     | Customrewiring | MSG    |
| Greedymodularity |                                    | 0.5901 |     | 0.6820         | 0.8094 |
| Labelpropagation |                                    | 0.6514 |     | 0.6039         | 0.8255 |
Girvan–Newman
|           |     | 0.4499 |     | 0.5964 | 0.9471 |
| --------- | --- | ------ | --- | ------ | ------ |
| AsynLPA   |     | 0.6911 |     | 0.6173 | 0.7152 |
| Bisection |     | 0.1437 |     | 0.1887 | 0.4152 |
| K_clique  |     | 0.2889 |     | 0.4739 | 0.6549 |
| Louvain   |     | 0.5901 |     | 0.6018 | 0.9511 |
Abbreviations:CDA,communitydetectionalgorithm;AsynLPA,asynchronouslabelpropagationalgorithm;MPL,
maximizationofpersistenceloss;MSG,maximizationofsafenessgain;NMI,NormalizedMutualInformation.

|
20 KUMARIETAL.
IntheMPLstrategy,theminimumandthemaximumvalueofNMIscoreis0.1437and0.6911
for Bisection and Asyn LPA CDAs, respectively. The average NMI score is 0.4864 for MPL
strategy.Forthecustomrewiringstrategy,theminimumandthemaximumvalueofNMIscore
is0.1887and0.6820forBisectionandGreedymodularityCDAs,respectively.Theaveragescore
of NMI is 0.5377 for custom rewiring strategy. For the MSG strategy, the minimum and the
maximum value of NMI score is 0.4152 and 0.9511 for Bisection and Louvain CDAs, respec-
tively.TheaveragescoreofNMIis0.7598forMSGstrategy.TheNMIscoreforallthediscussed
data sets under different strategies is analyzed. From the results, we can infer that the MPL
strategy is performing best, and MSG strategy is performing worst in terms of NMI of nodes
against CDAs in the network.
3.5 | Constant communities
CCsarethegroupofnodesthatalwayslietothesamecommunityformultipleCDAs.Weused
all the above‐mentioned CDAs and found the nodes in the CC are easily identifiable. Hence,
thepaperaims toreducethenumberofCCnodesandenhancesitsdeceptionscoreinsidethe
community. We proposed some strategies to destabilize the CC with minimum change in the
network which is tested on multiple data sets: Dolphins and Domestic Terror Network.
3.5.1 | Dolphins data set
The CCs of the Dolphins data set are detected using all the discussed seven CDAs (Figure 4).
ThenumberofnodesintheCCshownthroughPrussianbluecolor.Intheinitialnetwork,total
44 nodes are the part of different CCs where the total number of nodes in the network is 62.
The nodes that are part of the CCs is approximately 71%. In the MPL strategy, when we
performedtheupdateoperation,thenthenumberofnodesin theCCreducedto10. Now,the
nodes those part of the CC are approximately 16% only. In the custom rewiring approach,
the number of nodes in CC is 35 which is approximately 56% of the total number of nodes in
the data set. In the MSG approach, the number of nodes in CC is 41 which is approximately
66% of the network.
3.5.2 | Domestic terror network data set
The CCs of the Domestic terror network data set are detected using all the discussed seven
CDAs (Figure 4). The number of nodes in the CC shown through Prussian blue color. In the
initial network, a total of 22 nodes are the part of different CCs, where the total number of
nodes in the network is 32. The nodes that are part of the CCs are approximately 69%. In the
MPL strategy, when we performed the update operation, then the number of nodes in the CC
reducedto18.Now,thenodesthosepartoftheCCareapproximately56%only.IntheCustom
rewiring approach, the number of nodes in CC is 10, which is approximately 31% of the total
numberofnodesinthedataset.IntheMSGapproach,thenumberofnodesinCCis14,which
is approximately 44% of the network. From the discussed results on multiple data sets, it is
observedthatalltheproposedstrategiesarededucingthenumberofnodesinvolvedintheCCs.
For Dolphins data sets, MPL strategy is performing best and for Domestic terror network,

|
KUMARIETAL. 21
FIGURE 4 DolphinswithnumerousCCsdetectedthroughmultipleCDAsfor(A)initialnetwork,(B)network
afterMPLstrategy,(C)aftercustomrewiringoperation,and(D)afterapplyingMSGstrategy.Thenodesthatarenot
thepartofCCsareshowninPrussianbluecolor.ThenodesofthedifferentCCsareshownthroughdifferentcolors,
wherenodesofthesameCCareshownthroughthesamecolor.CC,constantcommunity;CDA,community
detectionalgorithm;MPL,maximizationofpersistenceloss;MSG,maximizationofsafenessgain[Colorfigurecanbe
viewedatwileyonlinelibrary.com]
customrewiringstrategyperformingbest.Incontrast,forallthedatasets,MSGstrategyhasthe
worst performance in terms of reduction of nodes involved in the CC (Figure 5).
3.6 | Detailed CC description
TheconstantcommunityCC
i
consistsofsomenodesNCCiandtheymaybeconnectedinternally
or with the nodes of some other communities. The total number of internal and external
connections is denoted by I(CC) and O(CC), respectively. The density ρ of each CC is
i i i
evaluated as ρ = I(CCi) , where ∣CC∣ is the number of nodes in CC. Larger value of
∣CCi∣(∣CCi∣−1) i i
ρ indicatesstronginternalconnectionsamongthenodesinCC.Theproposedstrategyshould
i
reduce the value of ρ so that the CCs can be easily destabilized. The hub dominance inside
CCi
eachCC is calculated as  = max .
i (∣CCi∣−1)
3.6.1 | Dolphins data set
ThedetailedinformationofCCsofDolphinsdatasetisprovidedinTable9.Initialnetwork
contains six CCs which consists of 32 nodes with 74 internal connections and 89 external

|
22 KUMARIETAL.
FIGURE 5 DomesticterrornetworkwithnumerousCCsdetectedthroughmultipleCDAsfor(A)initial
network,(B)afterMPLstrategy,(C)aftercustomrewiringoperation,and(D)afterapplyingMSGstrategy.The
nodesthatarenotthepartofCCsareshowninPrussianbluecolor.ThenodesofthedifferentCCsareshown
throughdifferentcolors,wherenodesofthesameCCareshownthroughthesamecolor.CC,constant
community;CDA,communitydetectionalgorithm;MPL,maximizationofpersistenceloss;MSG,maximization
ofsafenessgain[Colorfigurecanbeviewedatwileyonlinelibrary.com]
connections. The average density of the communities is 0.5360, and the average hub
dominance is 1.719. The network after MPL strategy contains two communities which
consistof10nodeswithoneinternal connectionand24externalconnections.Theaverage
density of each community is 0.0835 which is much less than the initial network and the
hub dominance is 1.600. The network after custom rewiring approach consists of five CCs
which contain 35 nodes with 58 internal connections and 84 external connections. The
averagedensityis0.5850and theaveragehub dominance scoreis2.133.The networkafter
application of MSG strategy consists of three CCs which contain 41 nodes with 88 internal
connections and 50 external connections. The average density is 0.3616, and the average
hub dominance is 0.9096. The order of average density is ρinitial > ρcustom > ρMSG > ρMPL.
The order of average hub dominance is custom > initial > MPL > MSG.
3.6.2 | Domestic terror network data set
The detailed information of CCs of Domestic terror network data set is provided in Table 10.
InitialnetworkcontainsthreeCCswhichconsistof22nodeswith30internalconnectionsand
14externalconnections.Theaveragedensityofthecommunitiesis0.4673,andtheaveragehub
dominanceis1.258.ThenetworkafterMPLstrategycontainsthreecommunitieswhichconsist
of 18 nodes with six internal connections and 42 external connections. The average density of

|
23
KUMARIETAL.
TABLE 9 ConstantcommunityinformationofDolphinsnetworkundermultiplestrategies
| Network  | CC  | N   | I(CC) | O(CC) | ρ     | ℋ     |
| -------- | --- | --- | ----- | ----- | ----- | ----- |
|          | i   |     | i     | i     |       |       |
|          | 1   | 7   | 13    | 26    | 0.619 | 2.000 |
|          | 2   | 5   | 2     | 18    | 0.200 | 2.250 |
|          | 3   | 5   | 8     | 11    | 0.800 | 1.500 |
| Initial  | 4   | 11  | 22    | 9     | 0.400 | 0.900 |
|          | 5   | 4   | 5     | 9     | 0.833 | 2.667 |
|          | 6   | 12  | 24    | 16    | 0.364 | 1.000 |
|          | 1   | 4   | 1     | 8     | 0.167 | 2.000 |
| MPL      | 2   | 6   | 0     | 16    | 0.000 | 1.200 |
|          | 1   | 16  | 35    | 9     | 0.292 | 0.600 |
|          | 2   | 6   | 10    | 28    | 0.667 | 2.400 |
| Custom   | 3   | 5   | 3     | 15    | 0.300 | 2.000 |
| Rewiring | 4   | 4   | 5     | 21    | 0.833 | 3.667 |
|          | 5   | 4   | 5     | 11    | 0.833 | 2.000 |
|          | 1   | 18  | 38    | 7     | 0.248 | 0.529 |
| MSG      | 2   | 11  | 26    | 27    | 0.473 | 1.200 |
|          | 3   | 12  | 24    | 16    | 0.364 | 1.000 |
Abbreviations:CC,constantcommunity;MPL,maximizationofpersistenceloss;MSG,maximizationofsafenessgain.
CC is 0.1356 which is much less than the initial network and the hub dominance is 1.262.
ThenetworkaftercustomrewiringapproachconsistsoftwoCCswhichcontain10nodeswith
nine internal connections and 13 external connections. The average density is 0.500, and the
average hubdominance score is 1.866. The network afterapplication of MSG strategy consists
TABLE 10 ConstantcommunityinformationofDomesticterrornetworkundermultiplestrategies
| Network  | CC i | N   | I(CC) i | O(CC) i | ρ     | ℋ     |
| -------- | ---- | --- | ------- | ------- | ----- | ----- |
|          | 1    | 7   | 11      | 5       | 0.524 | 1.500 |
| Initial  | 2    | 6   | 9       | 6       | 0.600 | 1.400 |
|          | 3    | 9   | 10      | 3       | 0.278 | 0.875 |
|          | 1    | 5   | 3       | 15      | 0.300 | 1.750 |
| MPL      | 2    | 5   | 0       | 11      | 0.000 | 0.750 |
|          | 3    | 8   | 3       | 16      | 0.107 | 1.286 |
| Custom   | 1    | 4   | 4       | 9       | 0.667 | 2.333 |
| Rewiring | 2    | 6   | 5       | 4       | 0.333 | 1.400 |
|          | 1    | 8   | 13      | 4       | 0.464 | 1.000 |
| MSG      | 2    | 6   | 9       | 7       | 0.600 | 1.800 |
Abbreviations:CC,constantcommunity;MPL,maximizationofpersistenceloss;MSG,maximizationofsafenessgain.

|
24
KUMARIETAL.
oftwoCCs whichcontain14nodes with22internalconnectionsand11external connections.
The average density is 0.532, and the average hub dominance is 1.400. The order of
average density is ρMSG > ρcustom > ρinitial > ρMPL. The order of average hub dominance
| is custom      | > MSG > MPL | > initial. |        |            |
| --------------- | ------------- | ----------- | ------ | ---------- |
| 4 | CONCLUSIONS |               | AND         | FUTURE | DIRECTIONS |
| 4.1 |           | Conclusions   |             |        |            |
Theoutcomeofthisstudyplaysavaluableroleinhidingtheidentityofnodesfromseveralaspects,
for example, protecting the identity of security agents in a terrorist network, hiding information of
individuals in a social network, and so forth. From the obtained result, we can observe the per-
centagegaininthesafenessscoreandpercentagelossinthepermanencescoreusingtheproposed
approachesagainstallthediscussedCDAs.TheNMIscoreforallthediscusseddatasetsunderthe
proposed strategies (MPL and custom rewiring) is less than the MSG strategy. From this, we can
inferthattheproposedstrategiesaremorecapableofhidingcommunityinformationthanthatofthe
MSGstrategy.Weanalyzedtheperformanceoftheproposedapproachesonmultiplereal‐worlddata
setsandfoundthatthenumberofnodesintheCCgotdecreasedfromtheinitialnetworkunderall
| the strategies: | MPL, custom       | rewiring, | and MSG approach. |     |
| --------------- | ----------------- | --------- | ----------------- | --- |
| 4.2 |           | Future directions |           |                   |     |
Inthispaper,sometargetnodesarekepthiddenfromCDAswiththehelpofsomegraph‐based
deceptionmechanisms(i.e.,clusteringcoefficient,maximumpullscore,reachability,etc.).The
MPLandtheMSGoperationsareperformedbyevaluatingthepersistencescoreofallthenodes
in the community structured network. But, with the help of a learning‐based mechanism, we
can evaluate the persistence score of some of the required nodes instead of calculating all the
nodes in the network. It reduces time complexity and rewiring cost. The current work can be
extended for a budget‐constrained rewiring operation. We may use a quantum mechanical
approach to hide nodes against adversarial attacks. Apart from that, the proposed deception
mechanism can be tested against some other real‐world data sets.
ACKNOWLEDGMENTS
This study is partially supported by the National Natural Science Foundation of China (Grant
No. 61872084).
ORCID
| Suchi Kumari | https://orcid.org/0000-0002-6748-5028 |     |     |     |
| ------------ | ------------------------------------- | --- | --- | --- |
Riteshkumar Jayprakash Yadav https://orcid.org/0000-0002-4093-1382
| Suyel Namasudra | https://orcid.org/0000-0002-0191-0175     |     |     |     |
| --------------- | ----------------------------------------- | --- | --- | --- |
| Ching‐Hsien     | Hsu https://orcid.org/0000-0002-2440-2771 |     |     |     |

|
KUMARIETAL. 25
REFERENCES
1. TahaK,YooPD.Usingthespanningtreeofacriminalnetworkforidentifyingitsleaders.IEEETransInf
ForensicsSecur.2017;12(2):445‐453.
2. PallaG,BarabásiAL,VicsekT.Quantifyingsocialgroupevolution.Nature.2007;446(7136):664‐667.
3. ChenL,QiaoS,HanN,etal.Friendshippredictionmodelbasedonfactorgraphsintegratinggeographical
location.CAAITransIntellTechnol.2020;5(3):193‐199.
4. LinY,WangYM.Decisionframeworkofgroupconsensuswithhesitantfuzzylinguisticpreferencerela-
tions.CAAITransIntellTechnol.2020;5(3):157‐164.
5. GuY,FuX,LiuZ,XuX,ChenA.Performanceoftransportationnetworkunderperturbations:reliability,
vulnerability,andresilience.TranspResPartE:LogisticsTranspRev.2020;133:101809.
6. ZengM,Zhang F,Wu FX,LiY,WangJ, LiM. Protein–proteininteraction siteprediction throughcom-
bininglocalandglobalfeatureswithdeepneuralnetworks.Bioinformatics.2020;36(4):1114‐1120.
7. Newman ME. Fast algorithm for detecting community structure in networks. Phys Rev E. 2004;69(6):
066133.
8. Blondel VD, Guillaume JL, Lambiotte R, Lefebvre E. Fast unfolding of communities in large networks.
JStatMech:TheoryExp.2008;2008(10):P10008.
9. LiJ,ZhangH,HanZ,RongY,ChengH,HuangJ.Adversarialattackoncommunitydetectionbyhiding
individuals. In: Proceedings of the Web Conference, New York, NY, USA: Association for Computing
Machinery; 2020:917‐927.https://doi.org/10.1145/3366423.3380171
10. Waniek M, Michalak TP, Wooldridge MJ, Rahwan T. Hiding individuals and communities in a social
network.NatHumBehav.2018;2(2):139.
11. KumariS,SinghA.Effectofcorrelationonthetrafficcapacityoftime‐varyingcommunicationnetwork.
ModPhysLettB.2019;33(26):1950314.
12. JiangR,MouX,ShiS,etal.Objecttrackingoneventcameraswithoffline–onlinelearning.CAAITrans
IntellTechnol;2020;5(3):165‐171.
13. White JG, Southgate E, Thomson JN, Brenner S. The structure of the nervous system of the nematode
Caenorhabditiselegans.PhilosTransRSocLondBBiolSci.1986;314(1165):1‐340.
14. TahaK,YooPD.SIIMCO:aforensicinvestigationtoolforidentifyingtheinfluentialmembersofacriminal
organization.IEEETransInfForensicsSecur.2015;11(4):811‐822.
15. Marshak CZ, Rombach MP, Bertozzi AL, D'Orsogna MR. Growth and containment of a hierarchical
criminalnetwork.PhysRevE.2016;93(2):022308.
16. Kumari S,Singh A. Time‐varying network modeling andits optimal routing strategy. Adv Complex Syst.
2018;21(02):1850006.
17. KumariS,SarohaA,SinghA.Efficientedgerewiringstrategiesforenhancementinnetworkcapacity.Phys
A:StatMechAppl.2020;545:123552.
18. NamasudraS.Cloudcomputing:anewera.JFundamApplSci.2018;10(2):113‐135.
19. NamasudraS,RoyP,BalamuruganB.Cloudcomputing:fundamentalsandresearchissues.In:Proceedings
of the Second International Conference on Recent Trends and Challenges in Computational Models
(ICRTCCM).Tindivanam,India:IEEE;2017.https://doi.org/10.1109/ICRTCCM.2017.49
20. Namasudra S, Chakraborty R, Majumder A, Moparthi NR. Securing multimedia by using DNA‐based
encryptioninthecloudcomputingenvironment.ACMTransMultimediaComputCommunAppl(TOMM).
2020;16(3s):1‐19.
21. NamasudraS.FastandsecuredataaccessingbyusingDNAcomputingforthecloudenvironment.IEEE
TransServComput.2020:1‐1.https://doi.org/10.1109/TSC.2020.3046471
22. Namasudra S,Deka GC. Applications of Blockchain in Healthcare. Singapore: Springer. 2021. https://doi.
org/10.1007/978-981-15-9547-9
23. Namasudra S, Chakraborty R, Kadry S, Manogaran G, Rawal BS. FAST: fast accessing scheme for data
transmission in cloud computing. Peer‐to‐Peer Networ Appl. 2020. https://doi.org/10.1007/s12083-020-
00959-6
24. Namasudra S. An improved attribute‐based encryption technique towards the data security in cloud
computing.ConcurrencyComput:PractExper.2019;31(3):e4364.

|
26 KUMARIETAL.
25. NamasudraS,SharmaS,DekaGC,LorenzP.DNAcomputingandtablebaseddataaccessinginthecloud
environment.JNetwComputAppl.2020;172:102835.
26. Newman ME. Modularity and community structure in networks. Proc Natl Acad Sci. 2006;103(23):
8577‐8582.
27. FortunatoS.Communitydetectioningraphs.PhysRep.2010;486(3‐5):75‐174.
28. ClausetA,NewmanME,MooreC.Findingcommunitystructureinverylargenetworks.PhysRevE.2004;
70(6):066111.
29. RaghavanUN,AlbertR,KumaraS.Nearlineartimealgorithmtodetectcommunitystructuresinlarge‐
scalenetworks.PhysRevE.2007;76(3):036106.
30. KernighanBW,LinS.Anefficientheuristicprocedureforpartitioninggraphs.BellSystTechJ.1970;49(2):
291‐307.
31. NagarajaS.Theimpactofunlinkabilityonadversarialcommunitydetection:effectsandcountermeasures.
In:AtallahM.J.,HopperN.J.(eds)PrivacyEnhancingTechnologies.PETS2010.LectureNotesinComputer
Science,Berlin,Heidelberg:Springer.2010;6205.https://doi.org/10.1007/978-3-642-14527-8_15
32. Zügner D, Borchert O, Akbarnejad A, Guennemann S. Adversarial attacks on graph neural networks:
perturbationsandtheirpatterns.ACMTransKnowlDiscoveryData(TKDD).2020;14(5):1‐31.
33. Zhu D, Zhang Z, Cui P, Zhu W. Robust graph convolutional networks against adversarial attacks. In:
ProceedingsoftheKDD'19:The25thACMSIGKDDConferenceonKnowledgeDiscoveryandDataMining,
AnchorageAKUSA.NewYork,NY,UnitedStates:AssociationforComputingMachinery;2019:1399‐1407.
34. JinW,LiY,XuH,WangY,TangJ.Adversarialattacksanddefensesongraphs:areviewandempirical
study.arXivpreprintarXiv:2003.00653;2020.
35. CaspiA,GorskyP.Onlinedeception:prevalence,motivation,andemotion.CyberPsycholBehav.2006;9(1):
54‐59.
36. WangW,ShangY,HeY,LiY,LiuJ.BotMark:automatedbotnetdetectionwithhybridanalysisofflow‐
basedandgraph‐basedtrafficbehaviors.InfSci.2020;511:284‐296.
37. Chen J, Chen L, Chen Y, et al. GA based Q‐attack on community detection. 2018. arXiv preprint
arXiv:1811.00430.
38. LiuY,LiuJ,ZhangZ,ZhuL,LiA.REM:fromstructuralentropytocommunitystructuredeception.In:H.
Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché‐Buc, E. Fox and R. Garne (eds.) Advances in Neural
InformationProcessingSystemsCurranAssociates,Inc.;2019;32:12938‐12948.
39. FiondaV,PirroG.Communitydeceptionor:howtostopfearingcommunitydetectionalgorithms.IEEE
TransKnowlDataEng.2018;30(4):660‐673.
40. ZhouY,ReidE,QinJ,ChenH,LaiG.USdomesticextremistgroupsontheWeb:linkandcontentanalysis.
IEEEIntellSyst.2005;20(5):44‐51.
41. LusseauD,SchneiderK,BoisseauOJ,HaaseP,SlootenE,DawsonSM.ThebottlenoseDolphincommunity
ofdoubtfulsoundfeaturesalargeproportionoflong‐lastingassociations.BehavEcolSociobiol.2003;54(4):
396‐405.
42. PedregosaF,VaroquauxG,GramfortA,etal.Scikit‐learn:machinelearninginPython.JMachLearnRes.
2011;12:2825‐2830.
How to cite this article: Kumari S, Yadav RJ, Namasudra S, Hsu C‐H. Intelligent
deceptiontechniquesagainstadversarialattackontheindustrialsystem.IntJIntellSyst.
2021;1‐26. https://doi.org/10.1002/int.22384
