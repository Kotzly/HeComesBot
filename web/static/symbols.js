import { SYM_SIZE } from './state.js';
import { colorParamToHex } from './utils.js';

export function symbolType(node) {
  if (node.arity === 1) return d3.symbolDiamond;
  if (node.arity === 2) return d3.symbolSquare;
  switch (node.func) {
    case 'rand_color': return d3.symbolCircle;
    case 'cone':       return d3.symbolTriangle;
    case 'circle':     return d3.symbolStar;
    case 'sphere':     return d3.symbolCircle;
    case 'x_var':      return d3.symbolCross;
    case 'y_var':      return d3.symbolWye;
    default:           return d3.symbolCircle;
  }
}

export function symbolPath(node) {
  return d3.symbol().type(symbolType(node)).size(SYM_SIZE)();
}

export function fillFor(node) {
  if (node.arity > 0) return node.arity === 1 ? '#569cd6' : '#ce9178';

  switch (node.func) {
    case 'rand_color': {
      const c = (node.params || {}).color || [0.5, 0.5, 0.5];
      return colorParamToHex(c);
    }
    case 'circle':
    case 'cone':
    case 'sphere':
    case 'x_var':
    case 'y_var': {
      const c = (node.params || {}).color;
      return c ? colorParamToHex(c) : '#c792ea';
    }
    default:
      return '#4ec9b0';
  }
}

export function strokeFor(node, isSelected) {
  if (isSelected) return {color: '#ffffff', width: 2.5};
  if (node.arity === 0 && (node.func === 'circle' || node.func === 'cone' || node.func === 'sphere')) {
    const p        = node.params || {};
    const areaFrac = Math.min(Math.PI * (p.rx || 0.5) * (p.ry || 0.5) / 4, 1);
    return {color: '#aaa', width: 1 + 5 * areaFrac};
  }
  return {color: '#333', width: 1};
}
