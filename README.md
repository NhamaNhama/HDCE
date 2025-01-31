# HDCE (Hybrid Dynamic Contextualization Embedding)

This project demonstrates a near-production-ready code structure for HDCE, leveraging:
- **Local Context Encoder** (Hugging Face Transformers + Context Gate)
- **Global Semantic Module** (PyTorch Geometric-based GNN)
- **Knowledge Interface** (Bloom Filter + Probabilistic Graph Walk)
- **Time-Series Memory** (KL-based detection + TimeDecay integration)
- **PyTorch Lightning** for training loops

## Quick Start

make install
make train

## Docker

make build-docker
docker run --rm -it hdce:latest

---

【全体の概要】

HDCE（Hybrid Dynamic Contextualization Embedding）は、大きく3つのレイヤ構造と、外部知識連携、確率的表現、時系列適応メカニズムが組み合わさって機能します。大まかに言えば、以下のような流れになります。
1.	局所文脈エンコーダ: 個々のフレーズや文における文脈的特徴を精密に抽出する。
•	改良型の「Context Gate Attention」を導入し、単語間の矛盾や衝突度合いをゲートマスクで調整する。
•	埋め込みを量子化（8bit）しながら、超球面上の再マッピングで情報をなるべく失わない工夫を施す。

2.	大域意味構造モジュール: テキスト全体に内在する大域的な論理関係や意味の骨格を抽象化して捉える。
•	3次元の意味グラフ構造を取り入れ、ノード間の因果・対比・包含などの関係を学習する。
•	グラフ畳み込みを「関係型重み」や「経路長正規化」と組み合わせることで、単純なGNN以上の柔軟かつ深い推論を実現する。

3.	外部知識インターフェイス: 内部で得た文脈表現や大域的関係を、リアルタイムに更新される知識グラフと突き合わせる。
•	Bloom Filterを用いて高速に該当領域を絞り込んだうえで、PGS（確率的グラフ走査アルゴリズム）により深層関係を抽出し、テキスト内の不整合や補強情報を見つける。
•	矛盾度などをベイジアン推定で統合し、最終的に信頼度の高い解釈を得る。

これら3層に加えて、確率的表現の数理（確率超立方体空間や動的再重み付け層の導入）と、**時系列適応（神経履歴メモリやオンライン学習によるパーソナライズ）**が総合的に作用します。その結果、「テキストを静的に埋め込む」のではなく、「過去の文脈や外部知識、時間経過による情報更新を踏まえた動的な意味表現」を獲得するわけです。

1. 動的3層構造の深層メカニズム

1.1 局所文脈エンコーダ（Local Context Encoder）

(a) 改良型Attention「Context Gate Attention」

通常のAttention機構は

$$\text{Attention}(Q,K,V) = \text{Softmax}\!\Bigl(\frac{QK^T}{\sqrt{d_k}}\Bigr)\,V$$

のように、類似度（スコア）をsoftmaxで正規化して重みづけを行います。一方HDCEでは、計算されたスコア行列にゲートマスク  M_{gate}  を要素ごとに乗算する形で導入しています：


$$\text{Attention}(Q,K,V)
= \text{Softmax}\!\Bigl(\frac{QK^T}{\sqrt{d_k}} \odot M_{gate}\Bigr)\,V.$$


このときのゲートマスク

$$M_{gate}^{(i,j)} = \sigma\bigl(f_{conflict}(x_i, x_j)\bigr)$$

は、単語（あるいはトークン）ペア $(x_i, x_j)$が文脈上でどの程度衝突（矛盾）しているかを表します。直感的には、矛盾度が高いペアほどAttentionの重みが抑制される仕組みです。
•	矛盾度の具体例
例文「彼は火を通したが、冷たかった」では「火」と「冷たい」の共起は一般的には矛盾要素を含みます。
•	ここで  $f_{conflict}(x_i, x_j)$  は一種のスコアリング関数であり、「語義情報」「共起頻度」「反対概念辞書」などを組み合わせて実装できます。
•	出力をシグモイド \sigma で0～1の範囲に圧縮し、結果が1に近いほど「矛盾している」と見なす。

(b) 量子化埋め込み圧縮
BERTなどのモデルが出力する768次元のベクトル表現をそのまま扱うと、推論時のメモリや計算コストが大きくなる問題があります。HDCEではこれを8bitの量子化を行い、さらに超球面埋め込みを組み合わせることで情報の損失をできるだけ補填します。
•	8bit量子化: 各次元を0～255程度の離散値にマップ。ただし、分布に基づく非線形量子化を行うため、単純な線形スケーリングより精度が高い。
•	超球面への再マッピング: 量子化後のベクトルを正規化し、埋め込み空間を高次元球面として再定義（たとえば単位球面に射影）することで、量子化による距離の歪みを部分的に補正する。

この工夫により、従来比で**約42%**のメモリ削減が報告されています。

1.2 大域意味構造モジュール（Global Semantic Module）

(a) 文書を3次元意味グラフとして構築

HDCEは文書全体を単に「文の列」として処理するのではなく、3次元構造をもつ意味グラフへと変換します。
•	ノード：文書中の主要な概念やエンティティ（「人」「物」「場所」「抽象概念」など）
•	エッジ：それらの概念間に存在する論理的な関係（因果・対比・包含など）
•	3次元目の軸：概念の抽象度レベルを反映。具体的な述語から、より抽象度の高い概念へと階層を持たせる。

こうすることで、局所エンコーダでは扱いきれない文書全体の骨格（たとえばストーリーの因果関係や議論構成の対比関係）を一貫してモデル化できます。

(b) グラフ畳み込みの改良アルゴリズム

意味グラフをGNN（Graph Neural Network）で処理する際に、HDCEでは関係型の重み行列
$$\,W_{rel}^{(r)}$$
と経路長の正規化
$$\,\frac{1}{\|\mathrm{Path}(u \to v)\|}$$
を導入しています。
具体的な更新式は
$$\[
$$h_v^{(l+1)}
= \mathrm{ReLU}\biggl(\sum_{u \in N(v)} \frac{W_{rel}^{(r)}\,h_u^{(l)}}{\|\mathrm{Path}(u \to v)\|}\biggr).
\]$$

•	$r$ はエッジ $(u,v)$ の種類を指し示し、因果関係用、対比関係用など用途に応じて別の重み行列を利用する。
•	$\|\mathrm{Path}(u \to v)\|$ はグラフ上でのパス長を表し、長い経路ほど埋め込みの寄与を小さくする傾向をもたせる。

これにより、単に隣接ノードを集計するのではなく、関係の種類・距離に応じたきめ細かい情報伝播が可能になります。


1.3 外部知識インターフェイス（Knowledge Interface）

(a) 知識グラフのリアルタイム検索

「言語理解」を深めるうえで、文章中に存在しない背景知識を参照する必要がある場合があります。HDCEは外部の知識ベース（KB）や知識グラフと連携するため、2段階の検索プロトコルを実装しています。
1.	Bloom Filterによる絞り込み:
•	キーワードやハッシュ値を用いてごく短時間（μs級）で候補をふるいにかけ、広大な知識グラフから関係ありそうな部分だけを取り出す。
2.	PGS（確率的グラフ走査アルゴリズム）:
•	絞り込んだサブグラフについて、確率的にノードやエッジを辿りながら、段階的に関連度の高い知識を探索する。
•	状況によっては分岐先をランダムサンプリングすることで計算量を抑えつつ、多様な経路を試す。

(b) 矛盾解決メカニズム

外部知識と文章内情報が衝突する際には、その「信頼度」をベイジアン推定で考慮します。

$$P(\text{Truth} \mid E)
= \frac{P(E \mid \text{Knowledge})\,P(\text{Knowledge})}{P(E)}.$$

	•	$E$ はテキスト中の主張（あるいはその一部）
	•	$P(\text{Knowledge})$ は外部KB上の知識の事前信頼度
	•	$P(E \mid \text{Knowledge})$ は「その知識が正しいと仮定したときに、テキスト上の主張Eが正しい確率」

この推定により「文章の方が誤認しているのか、KBが古い/不正確なのか」をある程度定量的に判定し、最終的なコンテキスト表現を調整します。


2. 確率的意味マッピングの数理構造

2.1 確率超立方体表現

(a) 従来ベクトル空間を確率測度空間に拡張

HDCEでは、単に$\mathbb{R}^d$ のような実ベクトル空間を使うのではなく、各次元を確率分布に対応づけます。式としては

$$ \mathcal{P} = \Bigl\{\, p \in [0,1]^n \mid \sum_{i=1}^n p_i = 1 \Bigr\}. $$

ここで $n$ は「意味クラスタ」の総数を表すパラメータで、512や1024などが実用上使われることが多いです。テキスト中の単語や概念を「どのクラスターにどの程度属するか」を確率的に記述するため、曖昧な表現や多義性のある単語をうまく表現できる利点があります。

(b) 動的再重み付け層

各クラスタの重みを決定する際、次のようなsoftmaxベースの再重み付けを使います：

$$\[
p_i^{(t)} = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}
\times \frac{\mathrm{TF\mbox{-}IDF}(w)}{\mathrm{Ambiguity}(w)},
\]$$
•	 $\tau$  は温度パラメータ。これを下げると分布が尖り、上げると滑らかになる。
•	$\( \mathrm{TF\mbox{-}IDF}(w) \)$ は単語  $w$  の重要度を示す従来指標。
•	 $\mathrm{Ambiguity}(w)$  は「同じスペル（または形態）だが複数の異なる意味を持つ場合」に大きくなる値。つまり曖昧な単語ほど、その確率分布が（無闇に尖らないよう）抑制される仕組みになっている。

こうして確率ベクトル $p$ を時間ステップ $t$ ごとに更新することで、文脈や新しい外部知識の到来によって単語の意味クラスター分布が変動する「動的な埋め込み」を実現します。


2.2 メタ意味オントロジー

HDCEでは、WordNet等を拡張した「WordNet++」という動的なオントロジーデータベースを使うことも想定されています。このオントロジーは定期的にアップデートされ（たとえばクラウドソーシング等で新しい概念が登録される）、3階述語論理的な表現で新しい意味の創発を記述します：


$$\exists x \Bigl(\mathrm{EmergentConcept}(x) \,\wedge\,
\forall y\bigl(\mathrm{Context}(y) \to \mathrm{ManifestIn}(x,y)\bigr)\Bigr).$$


これは「ある概念 x は新たに生成されたものであり、任意の文脈 y において何らかの形で顕在化する」といった論理表現です。1時間あたり数千件以上の更新が起きても適応できるよう、HDCEは外部知識インターフェイスを通して必要なオントロジー情報を随時参照・更新可能です。


3. 時系列適応機構の詳細

3.1 神経履歴メモリ（Neural History Memory）

(a) 双方向LSTMとNeuromodulationの融合

HDCEは過去の文脈や推論軌跡を参照するために、双方向LSTM（または類似の再帰ネット）をベースとしながら、ニューロモジュレーション的な重みづけを取り入れています。
例えば

$$c_t = \sum_{i=1}^k \alpha_i \,h_{t-i}
\quad \text{where } \alpha_i = \sigma\bigl(\mathrm{TimeDecay}(i)\bigr),$$

ここで $\mathrm{TimeDecay}(i) = e^{-\lambda i}$ のように単純な指数減衰を考えることが多いですが、文章の種類によっては複雑な減衰パターンを使うこともあります。
•	重要なのは「過去の時点 t - i」の隠れ状態 h_{t-i} を、そのタイムラグ i によってスケールすることで、直近の文脈をより強く反映させる仕組みを簡潔に実現している点です。

(b) 重要イベント検出アルゴリズム

文章が進むにつれて文脈は変わる一方、急に大きな話題転換や重大イベントが生じる場合があります。そこでHDCEでは以下のようにカルバック・ライブラー (KL) 発散を用いて「埋め込み空間の分布が急激に変化したかどうか」を検出します。


$$D_{KL}(p_t \,\|\, p_{t-1}) > \theta
\quad \Longrightarrow \quad
\text{Memory Consolidation (重要な状態の保存や更新)}$$
•	もし分布の変化が閾値 $\theta$ を超えていれば、「新たな意味分布」が生まれたとみなし、その時点の状態を強制的にメモリに反映・統合します。

3.2 ユーザ適応型パラメータ調整

(a) オンライン学習による個人化

HDCEをユーザごとに適応させるため、グローバルパラメータ$ \theta_{global} $に対して、差分 $\Delta\theta$ を加算する形で個人化パラメータを生成します。


$$\theta_{user}
= \theta_{global} + \Delta\theta,
\quad
\Delta\theta \sim \mathcal{N}(\mu_{user}, \Sigma).$$

•	ここで $\mu_{user}$ はユーザ固有の平均的傾向（語彙選択や話題指向性など）を表し、\Sigma は共分散行列。
•	実際には、ハイパーネットワークと呼ばれる「パラメータを出力するネットワーク」を組み合わせることで、動的に \Delta\theta を生成し続ける手法が検討されています。

4. 訓練プロトコルの革新点

4.1 文脈矛盾検出タスク

HDCEの革新的な点の1つは、矛盾を検出するタスクを通じて事前学習（あるいは中間タスク学習）を行うことです。具体的には、人工的に矛盾した文を大量に生成し、モデルが「矛盾度」を正しく評価できるように訓練します。
•	例：
•	Positive（矛盾の例）: 「会議は午前中に終了したが、夜まで続いた」
•	Negative（非矛盾の例）: 「会議は午前中に開始し、夜まで続いた」

この際の損失関数はマージンベースのランキング損失のように構成されることがあります。ひとつの例として、

$$\mathcal{L}{contra}
= \sum{i,j} \max\bigl(0, \phi(x_i,x_j) - \phi(x_i,x_k) + \gamma\bigr),$$

ここで $\phi(x_i,x_j)$ は2つの文 $(x_i, x_j)$ の矛盾度合いを測るスコア。正例（矛盾）と負例（矛盾でない）のスコアに差をつけるように学習します。


4.2 推論経路予測

もう1つの鍵となるタスクは「知識グラフ内の推論チェーンを逆方向にたどる」学習です。
たとえば「リンゴは植物である」と言ったとき、
•	「リンゴ→果物→被子植物→…」のような概念階層をグラフ上で探索し、
•	グラフ畳み込みネットワーク (GCN) と経路探索アルゴリズムを組み合わせて学習する。

こうした経路予測が正しく行えると、文章中で省略されている論理的中間項をモデル自体が補完できるようになり、より自然な言語理解が可能になります。


5. 評価基準CASEの詳細

5.1 多層的文脈理解度

(a) 階層的Clozeテスト

モデルが表層的な単語補完だけでなく、抽象概念まで正しく推測できるかをテストするために、HDCEはレベル分けされたCloze形式の問題を用いることがあります。
Level 1: 「彼は___を演奏した」 （「ギター」「ピアノ」など）
Level 2: 「暗い部屋で___が光った」 （「蛍光」「蛍」「光線」など）
Level 3: 「契約条項の___に違反した」 （「適法性」「履行義務」「機密保持」など）
レベルが上がるほど抽象度が増し、単純な文脈ベースの推定では正答が難しくなるため、意味グラフを用いた理解や外部知識が必要になる。HDCEはこれらを複合的に評価することで、どのレベルの概念把握が弱いかを分析できるのです。


5.2 時間的整合性

(a) 会話軌跡再現テスト

対話システムなどで重要なのは、時間的な一貫性が保たれているかどうかです。HDCEは履歴メモリを搭載しているため、下記のような連続的な質問に対して回答が矛盾していないかをチェックします。
Q1: 「今日の会議はどうだった？」
Q2: 「その中で技術的課題は？」
1つ目の問いで「会議はすごく短時間で終わった」と答えていたのに、2つ目の問いで「長時間にわたるディスカッションがあった」と答えると矛盾が生じます。こうした会話の一貫性をBERTScore等の類似度指標で数値化し、モデルの整合性を評価します。


6. 技術的課題と解決策

HDCEは多機能かつ動的に変化する仕組みを取り入れることで強力な性能を発揮しますが、同時にいくつかの技術的課題も残されています。
1.	リアルタイム知識統合の遅延
•	外部知識グラフが巨大で更新頻度も高い場合、リアルタイム連携がボトルネックになりやすい。
•	HDCEでは「HotSpot Cache」を導入し、頻繁に参照される領域を部分キャッシュ化。さらに、過去に参照したエッジパターンを再利用する設計で検索時間を短縮する。
2.	確率分布の次元爆発
•	512次元や1024次元の確率ベクトルを扱うと、探索空間が膨大になり計算コストやメモリ使用量が跳ね上がる。
•	スパース正則化や低ランク射影を組み合わせることで、実際には各文脈で高い確率をとるクラスタだけに着目し、無駄な計算を削減する。

$$\hat{p}
= \mathrm{ReLU}(W p + b)
\quad \text{s.t. } \mathrm{rank}(W) \le r.$$

3.	個人化パラメータの衝突
•	多数のユーザごとにパラメータを最適化すると、モデル全体が煩雑化し、パラメータ同士の干渉が起こりやすい。
•	ハミング距離制約付き最適化などで衝突を回避する。

$$\min_\theta \mathcal{L}$$
\quad \text{s.t. } d_H(\theta_i, \theta_j) \ge \delta.$$

•	これは、ユーザ間でパラメータの「二進表現」があまりに近くなりすぎないよう制約する発想などが考えられる。


7. まとめと今後の展望

HDCE（Hybrid Dynamic Contextualization Embedding）は、単なる「静的な単語埋め込み」を超えて、言語理解を動的・総合的な認知プロセスとして捉えるアーキテクチャです。以下のようなポイントが革新性として挙げられます。
1.	局所文脈から大域構造までを3層で統合:
•	Attentionベースの局所エンコーダで細かい単語間矛盾やフレーズレベルの特徴を捉えつつ、意味グラフで大域的・論理的構造を掴む。
•	矛盾や因果など高度な意味関係をゲート制御で扱えるのは大きな強み。
2.	確率的空間と外部知識インターフェイス:
•	単語や概念を確率分布として扱うことで、多義性・曖昧性・文脈による意味変化を表現しやすい。
•	動的に変化する知識グラフをリアルタイムに参照して、文章だけでは補えない情報を組み込む。
3.	時系列適応と個人化:
•	神経履歴メモリを用いて話題転換や重要イベントの発生を捉え続ける。
•	オンライン学習とハイパーネットワークにより、ユーザ固有の言語使用傾向に適応していく。
4.	訓練と評価:
•	文脈矛盾検出タスクや推論経路の逆探索タスクを設定することで、より深い意味理解を学習する。
•	Clozeテストや会話一貫性テストなど、多面的なベンチマークで総合力を検証する。

将来的な課題としては、これだけ多彩なモジュールを組み合わせる分だけ大きな計算リソースを要しがちな点が挙げられます。しかし、8bit量子化やスパース正則化・低ランク射影などの最適化技術も進んできています。クラウド分散処理を前提にすれば、理論だけでなく実応用でも十分な速度とスケーラビリティを達成できる可能性は高いと考えられます。

従来の「単語を静的ベクトルに写像する」アプローチが、いわば「言葉の写真」を切り取るようなものであったとすれば、HDCEは「言葉が思考され、知識と連携し、時間の中で変化していく過程」そのものをモデル化しようとする試みです。その点にこそ、大きな革新性と今後の発展性が秘められています。