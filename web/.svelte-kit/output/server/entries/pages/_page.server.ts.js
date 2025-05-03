const load = async ({ params }) => {
  return {
    experiments: [
      { id: 1, name: "Resnet", groups: ["Dogs", "Animals"], running: false, availableMetrics: ["loss", "val_loss"] },
      { id: 2, name: "DeezNuts3000", groups: ["Animals", "Cats"], running: true, availableMetrics: ["loss", "val_loss"] },
      { id: 3, name: "Hello, World", running: false, availableMetrics: ["loss", "val_loss"] },
      { id: 4, name: "Top Secret", groups: ["CIA"], running: true, availableMetrics: ["loss", "val_loss"] }
    ]
  };
};
const actions = {
  create: async ({ cookies, request }) => {
    console.log("creating new experiment...");
    const data = await request.formData();
    console.log(data);
  }
};
export {
  actions,
  load
};
