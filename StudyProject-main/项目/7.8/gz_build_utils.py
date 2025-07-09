import json

from py2neo import Graph
from tqdm import tqdm


class MedicalExtractor(object):
    def __init__(self):
        # 调用父类的构造函数
        super(MedicalExtractor, self).__init__()

        # 初始化图数据库连接
        self.graph = Graph("bolt://localhost:7687", auth=("neo4j", "153672"))
        # 初始化药品列表
        self.drugs = []
        # 初始化食物列表
        self.foods = []
        # 初始化疾病列表
        self.diseases = []
        # 初始化症状列表
        self.symptoms = []

        # 初始化禁止食用的食物列表
        self.rels_noteat = []
        # 初始化推荐食用的食物列表
        self.rels_doeat = []
        # 初始化推荐药物列表
        self.rels_recommanddrug = []
        # 初始化症状关系列表
        self.rels_symptom = []

    def extract_triples(self, data_path):
        print("从json中转换抽取三元组")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                # 读取每一行数据并解析为JSON格式
                data_json = json.loads(line)
                # 提取疾病名称并添加到疾病列表中
                disease = data_json['name']
                self.diseases.append(disease)
                # 检查是否有症状字段，如果有则处理症状数据
                if 'symptom' in data_json:
                    self.symptoms += data_json['symptom']
                    # 遍历症状列表，构建三元组并添加到关系列表中
                    for symptom in data_json['symptom']:
                        self.rels_symptom.append([disease, 'has_symptom', symptom])
                # 检查是否有伴随疾病字段，如果有则处理伴随疾病数据
                if 'acompany' in data_json:
                    for acompany in data_json['acompany']:
                        self.diseases.append(acompany)

                # 检查是否有推荐药物字段，如果有则处理推荐药物数据
                if 'recommand_drug' in data_json:
                    recommand_drug = data_json['recommand_drug']
                    self.drugs += recommand_drug
                    # 遍历推荐药物列表，构建三元组并添加到关系列表中
                    for drug in recommand_drug:
                        self.rels_recommanddrug.append([disease, 'recommand_drug', drug])

                # 检查是否有不宜食用字段，如果有则处理不宜食用数据
                if 'not_eat' in data_json:
                    not_eat = data_json['not_eat']
                    for _not in not_eat:
                        self.rels_noteat.append([disease, 'not_eat', _not])
                    self.foods += not_eat

                    # 检查是否有适宜食用字段，如果有则处理适宜食用数据
                    do_eat = data_json['do_eat']
                    for _do in do_eat:
                        self.rels_doeat.append([disease, 'do_eat', _do])
                    self.foods += do_eat

                    # 检查是否有药物详情字段，如果有则处理药物详情数据
                    if 'drug_detail' in data_json:
                        for det in data_json['drug_detail']:
                            det_spilt = det.split('(')
                            if len(det_spilt) == 2:
                                p, d = det_spilt
                                d = d.rstrip(')')
                                self.drugs.append(d)
                            else:
                                d = det_spilt[0]
                                self.drugs.append(d)


    def write_nodes(self, entitys, entity_type):
        print("写入 {0} 实体".format(entity_type))
        for node in tqdm(set(entitys), ncols=80):
            cql = """MERGE(n:{label}{{name:'{entity_name}'}})""".format(
                label=entity_type, entity_name=node.replace("'", ""))
            try:
                self.graph.run(cql)
            except Exception as e:
                print(e)
                print(cql)

    def create_entitys(self):
        self.write_nodes(self.drugs, '药品')
        self.write_nodes(self.symptoms, '菜谱')
        self.write_nodes(self.foods, '食物')
        self.write_nodes(self.diseases, '疾病')


    def write_edges(self,triples,head_type,tail_type):
        print("写入 {0} 关系".format(triples[0][1]))
        for head,relation,tail in tqdm(triples,ncols=80):
            cql = """MATCH(p:{head_type}),(q:{tail_type})
                    WHERE p.name='{head}' AND q.name='{tail}'
                    MERGE (p)-[r:{relation}]->(q)""".format(
                   head_type=head_type,tail_type=tail_type,head=head.replace("'",""),
                        tail=tail.replace("'",""),relation=relation)
            try:
                self.graph.run(cql)
            except Exception as e:
                print(e)
                print(cql)

    def create_relations(self):
        self.write_edges(self.rels_noteat, '疾病', '食物')
        self.write_edges(self.rels_doeat, '疾病', '食物')
        self.write_edges(self.rels_symptom, '疾病', '症状')
        self.write_edges(self.rels_recommanddrug, '疾病', '药品')


if __name__ == '__main__':
    extractor = MedicalExtractor()
    extractor.extract_triples('./graph_data/medical.json')
    extractor.create_entitys()
    extractor.create_relations()
