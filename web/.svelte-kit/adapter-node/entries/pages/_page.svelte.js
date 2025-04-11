import { l as current_component, m as rest_props, n as fallback, o as ensure_array_like, q as spread_attributes, t as clsx, u as element, h as slot, v as bind_props, f as pop, w as sanitize_props, p as push, x as spread_props, y as attr, k as escape_html } from "../../chunks/index.js";
import "../../chunks/client.js";
import "chart.js/auto";
function onDestroy(fn) {
  var context = (
    /** @type {Component} */
    current_component
  );
  (context.d ??= []).push(fn);
}
/**
 * @license lucide-svelte v0.469.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const defaultAttributes = {
  xmlns: "http://www.w3.org/2000/svg",
  width: 24,
  height: 24,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  "stroke-width": 2,
  "stroke-linecap": "round",
  "stroke-linejoin": "round"
};
function Icon($$payload, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const $$restProps = rest_props($$sanitized_props, [
    "name",
    "color",
    "size",
    "strokeWidth",
    "absoluteStrokeWidth",
    "iconNode"
  ]);
  push();
  let name = fallback($$props["name"], void 0);
  let color = fallback($$props["color"], "currentColor");
  let size = fallback($$props["size"], 24);
  let strokeWidth = fallback($$props["strokeWidth"], 2);
  let absoluteStrokeWidth = fallback($$props["absoluteStrokeWidth"], false);
  let iconNode = fallback($$props["iconNode"], () => [], true);
  const mergeClasses = (...classes) => classes.filter((className, index, array) => {
    return Boolean(className) && array.indexOf(className) === index;
  }).join(" ");
  const each_array = ensure_array_like(iconNode);
  $$payload.out += `<svg${spread_attributes(
    {
      ...defaultAttributes,
      ...$$restProps,
      width: size,
      height: size,
      stroke: color,
      "stroke-width": absoluteStrokeWidth ? Number(strokeWidth) * 24 / Number(size) : strokeWidth,
      class: clsx(mergeClasses("lucide-icon", "lucide", name ? `lucide-${name}` : "", $$sanitized_props.class))
    },
    void 0,
    void 0,
    3
  )}><!--[-->`;
  for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
    let [tag, attrs] = each_array[$$index];
    element($$payload, tag, () => {
      $$payload.out += `${spread_attributes({ ...attrs }, void 0, void 0, 3)}`;
    });
  }
  $$payload.out += `<!--]--><!---->`;
  slot($$payload, $$props, "default", {});
  $$payload.out += `<!----></svg>`;
  bind_props($$props, {
    name,
    color,
    size,
    strokeWidth,
    absoluteStrokeWidth,
    iconNode
  });
  pop();
}
function Maximize_2($$payload, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["polyline", { "points": "15 3 21 3 21 9" }],
    ["polyline", { "points": "9 21 3 21 3 15" }],
    [
      "line",
      {
        "x1": "21",
        "x2": "14",
        "y1": "3",
        "y2": "10"
      }
    ],
    [
      "line",
      {
        "x1": "3",
        "x2": "10",
        "y1": "21",
        "y2": "14"
      }
    ]
  ];
  Icon($$payload, spread_props([
    { name: "maximize-2" },
    $$sanitized_props,
    {
      iconNode,
      children: ($$payload2) => {
        $$payload2.out += `<!---->`;
        slot($$payload2, $$props, "default", {});
        $$payload2.out += `<!---->`;
      },
      $$slots: { default: true }
    }
  ]));
}
function Minimize_2($$payload, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    [
      "polyline",
      { "points": "4 14 10 14 10 20" }
    ],
    [
      "polyline",
      { "points": "20 10 14 10 14 4" }
    ],
    [
      "line",
      {
        "x1": "14",
        "x2": "21",
        "y1": "10",
        "y2": "3"
      }
    ],
    [
      "line",
      {
        "x1": "3",
        "x2": "10",
        "y1": "21",
        "y2": "14"
      }
    ]
  ];
  Icon($$payload, spread_props([
    { name: "minimize-2" },
    $$sanitized_props,
    {
      iconNode,
      children: ($$payload2) => {
        $$payload2.out += `<!---->`;
        slot($$payload2, $$props, "default", {});
        $$payload2.out += `<!---->`;
      },
      $$slots: { default: true }
    }
  ]));
}
function Interactive_chart($$payload, $$props) {
  push();
  onDestroy(() => {
    console.log("destroying chart!");
  });
  function destroy() {
  }
  $$payload.out += `<canvas id="myChart"></canvas>`;
  bind_props($$props, { destroy });
  pop();
}
function Experiments_list($$payload, $$props) {
  let { experiments } = $$props;
  let expandedId = null;
  const each_array = ensure_array_like(experiments);
  $$payload.out += `<section><div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"><!--[-->`;
  for (let $$index_2 = 0, $$length = each_array.length; $$index_2 < $$length; $$index_2++) {
    let experiment = each_array[$$index_2];
    $$payload.out += `<div${attr("class", `
                    bg-white rounded-sm p-4 
                    ${expandedId === experiment.id ? "md:col-span-2 lg:col-span-4 row-span-2 order-first" : "order-none"}
                `)}><article${attr("class", "flex flex-col gap-1")}>`;
    if (expandedId !== experiment.id) {
      $$payload.out += "<!--[-->";
      $$payload.out += `<div class="flex flex-row justify-between"><h3 class="font-medium text-lg text-gray-900">${escape_html(experiment.name)}</h3> <button>`;
      Maximize_2($$payload, {
        class: "w-5 h-5 text-gray-400 hover:text-gray-600"
      });
      $$payload.out += `<!----></button></div> <p class="text-gray-400 text-sm leading-relaxed">Lorem ipsum odor amet, consectetuer adipiscing elit.</p> <div class="flex flex-row gap-1 text-sm text-gray-500"><span>Groups:</span> `;
      if (experiment?.groups) {
        $$payload.out += "<!--[-->";
        const each_array_1 = ensure_array_like(experiment.groups);
        $$payload.out += `<ul class="flex flex-row gap-1"><!--[-->`;
        for (let $$index = 0, $$length2 = each_array_1.length; $$index < $$length2; $$index++) {
          let group = each_array_1[$$index];
          $$payload.out += `<li class="items-center"><span>${escape_html(group)}</span></li>`;
        }
        $$payload.out += `<!--]--></ul>`;
      } else {
        $$payload.out += "<!--[!-->";
      }
      $$payload.out += `<!--]--></div> <div class="text-sm"><span class="text-gray-500">Status:</span> <span${attr("class", `
                                    ${experiment.running ? "text-sky-400" : "text-orange-400"}
                                    `)}>${escape_html(experiment.running ? "Running" : "Stopped")}</span></div> <time class="text-gray-300 text-xs mt-auto pt-4">Created: 00-00-0000: 00:00</time>`;
    } else {
      $$payload.out += "<!--[!-->";
      $$payload.out += `<div class="flex flex-row justify-between"><h3 class="font-medium text-lg text-gray-900 text-left">${escape_html(experiment.name)}</h3> <button>`;
      Minimize_2($$payload, {
        class: "w-5 h-5 text-gray-400 hover:text-gray-600"
      });
      $$payload.out += `<!----></button></div> <div class="text-gray-400 leading-relaxed"><p>Lorem ipsum odor amet, consectetuer adipiscing
                                elit.</p></div> <div class="flex flex-row gap-1 text-gray-500"><span>Groups:</span> `;
      if (experiment?.groups) {
        $$payload.out += "<!--[-->";
        const each_array_2 = ensure_array_like(experiment.groups);
        $$payload.out += `<ul class="flex flex-row gap-1"><!--[-->`;
        for (let $$index_1 = 0, $$length2 = each_array_2.length; $$index_1 < $$length2; $$index_1++) {
          let group = each_array_2[$$index_1];
          $$payload.out += `<li class="items-center"><span>${escape_html(group)}</span></li>`;
        }
        $$payload.out += `<!--]--></ul>`;
      } else {
        $$payload.out += "<!--[!-->";
      }
      $$payload.out += `<!--]--></div> `;
      Interactive_chart($$payload, {});
      $$payload.out += `<!---->`;
    }
    $$payload.out += `<!--]--></article></div>`;
  }
  $$payload.out += `<!--]--></div></section>`;
}
function _page($$payload, $$props) {
  push();
  let { data } = $$props;
  let experiments = data.experiments;
  {
    $$payload.out += "<!--[!-->";
  }
  $$payload.out += `<!--]--> <header><nav class="px-6 py-4 flex flex-row justify-end bg-white border-b border-gray-200"><button class="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-sm transition-colors">(+) Experiment</button></nav></header> <main class="mx-4 my-4">`;
  Experiments_list($$payload, { experiments });
  $$payload.out += `<!----></main>`;
  pop();
}
export {
  _page as default
};
