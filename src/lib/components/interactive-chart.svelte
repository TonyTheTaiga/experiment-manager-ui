<script lang="ts">
    import chartjs from "chart.js/auto";
    let chartData;
    import { onMount } from "svelte";

    let chartValues = [20, 10, 5, 2, 20, 30, 45];
    let chartLabels = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
    ];
    let ctx;
    let chartCanvas;

    onMount(() => {
        ctx = chartCanvas.getContext("2d");
        var chart = new chartjs(ctx, {
            type: "line",
            data: {
                labels: chartLabels,
                datasets: [
                    {
                        label: "Revenue",
                        backgroundColor: "rgb(255, 99, 132)",
                        borderColor: "rgb(255, 99, 132)",
                        data: chartValues,
                    },
                ],
            },
        });
        return () => {
            chartCanvas.destroy();
        };
    });
</script>

<canvas bind:this={chartCanvas} id="myChart"></canvas>
