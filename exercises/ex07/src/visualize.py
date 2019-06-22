import sys
from model_search import Genotype
from graphviz import Digraph


def plot(genotype, filename):
  g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  #g.node("c_{k-2}", fillcolor='darkseagreen2')
  g.node("c_{k-1}", fillcolor='darkseagreen2')
  #assert len(genotype) % 2 == 0
  steps = 2

  for i in range(steps):
    g.node(str(i), fillcolor='lightblue')

  g.edge("c_{k-1}", str(0), label=genotype[0][0], fillcolor="gray")
  g.edge("c_{k-1}", str(1), label=genotype[2][0], fillcolor="gray")
  g.edge(str(0), str(1), label=genotype[1][0], fillcolor="gray")

  g.node("c_{k}", fillcolor='palegoldenrod')
  for i in range(steps):
    g.edge(str(i), "c_{k}", fillcolor="gray")

  g.render(filename, view=True)


if __name__ == '__main__':
  #genotype_name = sys.argv[1]
  with open('logs/architecture', 'r') as f:
      genotype = f.readline()
  try:
    genotype = eval(genotype)
  except AttributeError:
    print("{} is not specified in genotypes.py".format(genotype_name)) 
    sys.exit(1)

  plot(genotype.normal, "normal")
  plot(genotype.reduce, "reduction")

