import { createBrowserRouter } from "react-router";
import { LandingScreen } from "./components/LandingScreen";
import { LoadingScreen } from "./components/LoadingScreen";
import { DashboardScreen } from "./components/DashboardScreen";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: LandingScreen,
  },
  {
    path: "/processing",
    Component: LoadingScreen,
  },
  {
    path: "/results",
    Component: DashboardScreen,
  },
]);
