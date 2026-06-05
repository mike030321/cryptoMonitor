import { ReactNode, useState, useEffect, useCallback } from "react";
import { Link, useLocation } from "wouter";
import { Activity, Brain, LineChart, Coins, LayoutDashboard, Sparkles, Beaker, Cpu, Stethoscope, Archive, Menu, X, DollarSign } from "lucide-react";
import { cn } from "@/lib/utils";
import { ContactFormDialog } from "@/components/contact-form-dialog";

// Task #512 — `Archived Agents` lives at the bottom of the nav as a
// secondary link. The 4 deterministic executor families are exposed
// from the dashboard's Family Fleet card cluster, not as nav entries
// (the operator clicks a card to drill in).
const NAV_ITEMS = [
  { href: "/", label: "Command Center", icon: LayoutDashboard, emoji: "🎯" },
  { href: "/agents", label: "AI Agents", icon: Brain, emoji: "🧠" },
  { href: "/predictions", label: "Live Feed", icon: Activity, emoji: "⚡" },
  { href: "/analytics", label: "Analytics", icon: LineChart, emoji: "📊" },
  { href: "/coins", label: "Markets", icon: Coins, emoji: "💎" },
  { href: "/lab", label: "Strategy Lab", icon: Beaker, emoji: "🧪" },
  { href: "/shadow", label: "Quant Live Health", icon: Cpu, emoji: "🤖" },
  { href: "/diagnostics", label: "Brain Diagnostics", icon: Stethoscope, emoji: "🩺" },
  { href: "/agents/archived", label: "Archived Agents", icon: Archive, emoji: "🗄️" },
  { href: "/price-list", label: "Price List", icon: DollarSign, emoji: "💰" },
];

export function Layout({ children }: { children: ReactNode }) {
  const [location, navigate] = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Intercept all clicks when NOT on /price-list → redirect there
  // On /price-list everything works freely (buttons open form, etc.)
  const handleClick = useCallback((e: MouseEvent) => {
    if (location === "/price-list") return;
    const t = e.target as Element;
    if (t.closest('[role="dialog"]')) return;
    if (t.closest('.react-international-phone-country-selector-dropdown')) return;
    e.stopPropagation();
    e.preventDefault();
    navigate("/price-list");
  }, [location, navigate]);

  useEffect(() => {
    document.addEventListener("click", handleClick, true);
    return () => document.removeEventListener("click", handleClick, true);
  }, [handleClick]);

  return (
    <div className="fixed inset-0 bg-background text-foreground flex flex-row overflow-hidden">
      {/* Mobile backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={cn(
        "fixed inset-y-0 left-0 w-72 border-r border-sidebar-border bg-sidebar/70 backdrop-blur-xl flex-shrink-0 flex flex-col z-50 transition-transform duration-300",
        "md:static md:translate-x-0 md:z-10",
        sidebarOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0",
      )}>
        <div className="p-6 flex items-center gap-3">
          <div className="w-11 h-11 rounded-2xl animated-gradient flex items-center justify-center shadow-lg shadow-primary/30 ring-1 ring-white/20">
            <Sparkles className="w-5 h-5 text-white drop-shadow" />
          </div>
          <div className="flex-1">
            <h1 className="font-display font-bold text-xl tracking-tight gradient-text">Nexus</h1>
            <div className="text-[11px] font-medium text-muted-foreground flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 live-dot" />
              Live trading agents
            </div>
          </div>
          {/* Close button — mobile only */}
          <button
            className="md:hidden p-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-muted-foreground hover:text-foreground transition-colors"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close sidebar"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <nav className="flex-1 overflow-y-auto py-2 px-3">
          <div className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/70 px-3 mb-2">
            Navigate
          </div>
          <ul className="space-y-1">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = location === item.href || (item.href !== "/" && location.startsWith(item.href));

              return (
                <li key={item.href}>
                  <Link href={item.href} onClick={() => setSidebarOpen(false)}>
                    <div
                      className={cn(
                        "flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 cursor-pointer group relative",
                        isActive
                          ? "bg-gradient-to-r from-primary/20 to-secondary/10 text-foreground ring-1 ring-primary/30 shadow-lg shadow-primary/10"
                          : "text-muted-foreground hover:text-foreground hover:bg-white/5"
                      )}
                      data-testid={`nav-${item.label.toLowerCase().replace(/\s+/g, "-")}`}
                    >
                      <span className={cn(
                        "w-8 h-8 rounded-lg flex items-center justify-center transition-all",
                        isActive
                          ? "bg-gradient-to-br from-primary/40 to-secondary/30 text-white shadow"
                          : "bg-white/5 text-muted-foreground group-hover:bg-white/10 group-hover:text-foreground"
                      )}>
                        <Icon className="w-4 h-4" />
                      </span>
                      <span className="flex-1">{item.label}</span>
                      {isActive && <span className="w-1.5 h-1.5 rounded-full bg-primary live-dot" />}
                    </div>
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>

        {/* Contact button */}
        <div className="px-3 pb-2">
          <ContactFormDialog onCloseSidebar={() => setSidebarOpen(false)} />
        </div>

        <div className="p-4 m-3 rounded-2xl bg-gradient-to-br from-primary/10 via-secondary/5 to-transparent border border-primary/15">
          <div className="flex items-center justify-between text-xs">
            <span className="font-mono text-muted-foreground">v2.4.1</span>
            <span className="flex items-center gap-1.5 text-emerald-400 font-medium">
              <span className="w-2 h-2 rounded-full bg-emerald-400 live-dot" />
              Connected
            </span>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-h-0 min-w-0 overflow-hidden z-10">
        {/* Mobile top bar with hamburger */}
        <div className="md:hidden flex items-center gap-3 px-4 py-3 border-b border-border bg-background/80 backdrop-blur-sm flex-shrink-0">
          <button
            onClick={() => setSidebarOpen(true)}
            className="p-2 rounded-xl bg-white/5 hover:bg-white/10 text-muted-foreground hover:text-foreground transition-colors"
            aria-label="Open sidebar"
          >
            <Menu className="w-5 h-5" />
          </button>
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg animated-gradient flex items-center justify-center shadow-md shadow-primary/30">
              <Sparkles className="w-3.5 h-3.5 text-white" />
            </div>
            <span className="font-display font-bold text-base gradient-text">Nexus</span>
          </div>
        </div>

        <div className="flex-1 min-h-0 min-w-0 overflow-y-scroll overflow-x-hidden p-4 md:p-8" style={{ scrollbarGutter: "stable" }}>
          <div className="w-full">
            {children}
          </div>
        </div>
      </main>

      {/* Ambient background glow blobs */}
      <div className="fixed top-[-15%] left-[-10%] w-[60%] h-[60%] rounded-full bg-primary/15 blur-[140px] pointer-events-none z-0 animate-pulse" style={{ animationDuration: "8s" }} />
      <div className="fixed bottom-[-20%] right-[-10%] w-[55%] h-[55%] rounded-full bg-secondary/12 blur-[140px] pointer-events-none z-0 animate-pulse" style={{ animationDuration: "10s" }} />
      <div className="fixed top-[40%] right-[20%] w-[30%] h-[30%] rounded-full bg-cyan-500/8 blur-[120px] pointer-events-none z-0" />
    </div>
  );
}
