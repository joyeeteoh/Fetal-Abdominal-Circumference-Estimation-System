import "../styles/LandingPage.css";

import HeroSection from "../components/HeroSection";
import AboutSection from "../components/AboutSection";
import FeaturesSection from "../components/FeaturesSection";
import InfoTilesSection from "../components/InfoTilesSection";
import Footer from "../components/Footer";

export default function Home() {
  return (
    <>
      <HeroSection />
      <AboutSection />
      <FeaturesSection />
      <InfoTilesSection />
      <Footer />
    </>
  );
}
