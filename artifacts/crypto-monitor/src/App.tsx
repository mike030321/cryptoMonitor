import { useEffect } from "react";
import { Switch, Route, Router as WouterRouter, useLocation } from "wouter";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Layout } from "@/components/layout";

import Dashboard from "@/pages/dashboard";
import PriceList from "@/pages/price-list";
import Agents from "@/pages/agents";
import AgentDetail from "@/pages/agent-detail";
// Task #512 — executor-fleet drill-down + archived legacy bots.
import FamilyDrillDown from "@/pages/family-drill-down";
import ArchivedAgents from "@/pages/archived-agents";
import Predictions from "@/pages/predictions";
import Analytics from "@/pages/analytics";
import Coins from "@/pages/coins";
import CoinDetail from "@/pages/coin-detail";
import StrategyLab from "@/pages/strategy-lab";
import QuantShadow from "@/pages/quant-shadow";
import AdminPanel from "@/pages/admin";
import Diagnostics from "@/pages/diagnostics";
import NotFound from "@/pages/not-found";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function Router() {
  return (
    <Layout>
      <Switch>
        <Route path="/" component={Dashboard} />
        <Route path="/price-list" component={PriceList} />
        <Route path="/agents" component={Agents} />
        {/* Task #512 — keep these BEFORE the catch-all `/agents/:id`
            so the more-specific paths win the wouter match. */}
        <Route path="/agents/archived" component={ArchivedAgents} />
        <Route path="/agents/families/:profileId" component={FamilyDrillDown} />
        <Route path="/agents/:id" component={AgentDetail} />
        <Route path="/predictions" component={Predictions} />
        <Route path="/analytics" component={Analytics} />
        <Route path="/coins" component={Coins} />
        <Route path="/coins/:id" component={CoinDetail} />
        <Route path="/lab" component={StrategyLab} />
        <Route path="/shadow" component={QuantShadow} />
        <Route path="/admin" component={AdminPanel} />
        <Route path="/diagnostics" component={Diagnostics} />
        <Route component={NotFound} />
      </Switch>
    </Layout>
  );
}

function AppInner() {
  const [, navigate] = useLocation();

  // On every page load / hard refresh → always start at Command Center
  useEffect(() => {
    navigate("/", { replace: true });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return <Router />;
}

function App() {
  // Enforce dark mode on body
  document.documentElement.classList.add("dark");
  
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, "")}>
          <AppInner />
        </WouterRouter>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
