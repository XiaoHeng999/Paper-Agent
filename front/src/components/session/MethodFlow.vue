<script setup lang="ts">
import { computed, useId } from 'vue';
import dagre from '@dagrejs/dagre';
import type { FlowEdge } from '../../types/research';

const props = defineProps<{ edges: FlowEdge[]; paper: string; partial?: boolean }>();
const emit = defineEmits<{ evidence: [ids: string[]] }>();
const arrow = useId();
const wrap = (text: string, length: number) => Array.from(text).reduce<string[]>((lines, c, i) => {
  if (i % length === 0) lines.push('');
  lines[lines.length - 1] += c;
  return lines;
}, []);
const drawing = computed(() => {
  const graph = new dagre.graphlib.Graph({ multigraph: true }).setGraph({
    rankdir: 'TB', nodesep: 24, ranksep: 34, marginx: 15, marginy: 15
  });
  const names = [...new Set(props.edges.flatMap(e => [e.source, e.target]))];
  names.forEach((name, i) => graph.setNode(String(i), {width: 168, height: 12 + wrap(name, 10).length * 19}));
  props.edges.forEach((e, i) => graph.setEdge(String(names.indexOf(e.source)), String(names.indexOf(e.target)),
    {width: 130, height: wrap(e.transfer, 9).length * 17 + 6}, String(i)));
  dagre.layout(graph);
  return {
    size: graph.graph(),
    nodes: names.map((name, i) => ({...graph.node(String(i)), lines: wrap(name, 10)})),
    edges: graph.edges().map(edge => {
      const index = Number(edge.name);
      const positioned = graph.edge(edge) as dagre.GraphEdge & { x: number; y: number };
      return {index, x: positioned.x, y: positioned.y, lines: wrap(props.edges[index].transfer, 9),
        points: positioned.points.map(p => p.x + ',' + p.y).join(' ')};
    })
  };
});
</script>
<template>
  <figure>
    <figcaption><strong>{{ partial ? "局部数据流" : "算法数据流" }} · 系统教学重绘</strong></figcaption>
    <div class="canvas">
      <svg :width="drawing.size.width" :height="drawing.size.height" role="img" aria-label="算法数据流，逐条说明见图下">
        <defs><marker :id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#426674"/>
        </marker></defs>
        <polyline v-for="e in drawing.edges" :key="e.index" :points="e.points" fill="none" stroke="#426674" stroke-width="2" :marker-end="'url(#' + arrow + ')'"/>
        <g v-for="e in drawing.edges" :key="'label-'+e.index">
          <text v-for="(line, i) in e.lines" :key="i" :x="e.x" :y="e.y + (i - (e.lines.length - 1)/2)*17" class="edge">{{ line }}</text>
        </g>
        <g v-for="(n, index) in drawing.nodes" :key="index">
          <rect :x="n.x-n.width/2" :y="n.y-n.height/2" :width="n.width" :height="n.height" rx="7" fill="#e8f2f1" stroke="#54827f"/>
          <text v-for="(line,i) in n.lines" :key="i" :x="n.x" :y="n.y + (i - (n.lines.length - 1)/2)*19">{{ line }}</text>
        </g>
      </svg>
    </div>
    <ol><li v-for="(edge, i) in edges" :id="paper + '-flow-' + i" :key="i">
      {{ edge.source }} → {{ edge.target }} · {{ edge.transfer }}
      <button type="button" @click="emit('evidence', edge.evidence_ids)">查看依据</button>
    </li></ol>
  </figure>
</template>
<style scoped>
figure { margin: 1rem 0; }
.canvas { overflow: auto; background: #fafcfb; }
text { text-anchor: middle; dominant-baseline: central; font: 14px sans-serif; fill: #173638; }
.edge { font-size: 13px; paint-order: stroke; stroke: #fafcfb; stroke-width: 4px; }
li:target { outline: 2px solid #54827f; }
</style>
