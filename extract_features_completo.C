#include "Riostream.h" // Para cout, endl
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TMath.h"
#include <vector>
#include <string>
#include <iostream>
#include <TCanvas.h>
#include <TH2D.h>

void extract_features_completo() {

    // --- Configuración ---
    //  Cambiar nombre para los distintos tipos de partícula
    const char* output_filename = "extracted_features_all.root";
    const char* data_input_filename = "data_production/CLOUD_ALL_fill_FP_parallel_16mm_(11-20).ntuple.root";

    // Abrir archivo de entrada y obtener el árbol
    TFile *inputFile = TFile::Open(data_input_filename, "READ");
    if (!inputFile || inputFile->IsZombie()) {
        std::cerr << "Error: No se pudo abrir el archivo de entrada: " << data_input_filename << std::endl;
        return;
    }

    TTree *inputTree = (TTree*)inputFile->Get("output");
    if (!inputTree) {
        std::cerr << "Error: No se pudo encontrar el árbol 'output' en el archivo: " << data_input_filename << std::endl;
        inputFile->Close();
        return;
    }

    // Crear archivo de salida
    TFile *outputFile = new TFile(output_filename, "RECREATE");
    if (!outputFile || outputFile->IsZombie()) {
        std::cerr << "Error: No se pudo crear el archivo de salida: " << output_filename << std::endl;
        inputFile->Close();
        return;
    }
    
    // Crear un directorio para almacenar los histogramas
    TDirectory *histoDir = outputFile->mkdir("histogramas");
    if (!histoDir) {
        std::cerr << "Error: No se pudo crear el directorio para histogramas" << std::endl;
        outputFile->Close();
        inputFile->Close();
        return;
    }
    
    // Crear el árbol de características
    TTree *featureTree = new TTree("feature_tree", "Tree with calculated features");
    
    // Variables para el árbol de características
    Float_t Ec1, Ec2, Qmx, QmxP, QmxM, tminP, tminM, dp, Qmx2_val, Qtot_p, Qtot_m;
    Int_t nhits_fired_sipm;
    Int_t event_number;  // Para mantener referencia al evento original
    Int_t mcpdg;        // Etiqueta del evento (PDG code)

    // Añadir ramas al árbol de características
    featureTree->Branch("event_number", &event_number, "event_number/I");
    featureTree->Branch("mcpdg", &mcpdg, "mcpdg/I");
    featureTree->Branch("Ec1", &Ec1, "Ec1/F");
    featureTree->Branch("Ec2", &Ec2, "Ec2/F");
    featureTree->Branch("Qmx", &Qmx, "Qmx/F");
    featureTree->Branch("QmxP", &QmxP, "QmxP/F");
    featureTree->Branch("QmxM", &QmxM, "QmxM/F");
    featureTree->Branch("tminP", &tminP, "tminP/F");
    featureTree->Branch("tminM", &tminM, "tminM/F");
    featureTree->Branch("dp", &dp, "dp/F");
    featureTree->Branch("Qmx2_val", &Qmx2_val, "Qmx2_val/F");
    featureTree->Branch("Qtot_p", &Qtot_p, "Qtot_p/F");
    featureTree->Branch("Qtot_m", &Qtot_m, "Qtot_m/F");
    featureTree->Branch("nhits_fired_sipm", &nhits_fired_sipm, "nhits_fired_sipm/I");
    
    // Clonar el árbol original completo
    TTree *outputTree = inputTree->CloneTree(-1, "fast");  // Clonar todos los eventos
    outputTree->SetName("output");  // Mantener el mismo nombre que el árbol de entrada

    // --- Configurar ramas de entrada ---
    // Variables del árbol de entrada
    std::vector<int> *mcPMTID = nullptr;
    std::vector<int> *mcPMTNPE = nullptr;
    std::vector<double> *mcPEHitTime = nullptr;
    std::vector<double> *mcPEx = nullptr;
    std::vector<double> *mcPEy = nullptr;
    std::vector<double> *mcPEz = nullptr;
    std::vector<double> *mcPEFrontEndTime = nullptr;
    std::vector<int> *mcPEProcess = nullptr;
    std::vector<double> *mcPEWavelength = nullptr;
    std::vector<int> *trackPDG = nullptr;
    std::vector<std::vector<double>> *trackPosX = nullptr;
    std::vector<std::vector<double>> *trackPosY = nullptr;
    std::vector<std::vector<double>> *trackPosZ = nullptr;
    std::vector<std::vector<double>> *trackMomX = nullptr;
    std::vector<std::vector<double>> *trackMomY = nullptr;
    std::vector<std::vector<double>> *trackMomZ = nullptr;
    std::vector<std::vector<double>> *trackKE = nullptr;
    std::vector<std::vector<double>> *trackTime = nullptr;
    std::vector<std::vector<int>> *trackProcess = nullptr;
    Int_t mcpdg_input;
    Double_t mcke_input;

    // Configurar direcciones de las ramas
    inputTree->SetBranchAddress("mcpdg", &mcpdg_input);
    inputTree->SetBranchAddress("mcke", &mcke_input);
    inputTree->SetBranchAddress("mcPMTID", &mcPMTID);
    inputTree->SetBranchAddress("mcPMTNPE", &mcPMTNPE);
    inputTree->SetBranchAddress("mcPEHitTime", &mcPEHitTime);
    inputTree->SetBranchAddress("mcPEx", &mcPEx);
    inputTree->SetBranchAddress("mcPEy", &mcPEy);
    inputTree->SetBranchAddress("mcPEz", &mcPEz);
    inputTree->SetBranchAddress("mcPEFrontEndTime", &mcPEFrontEndTime);
    inputTree->SetBranchAddress("mcPEProcess", &mcPEProcess);
    inputTree->SetBranchAddress("mcPEWavelength", &mcPEWavelength);
    inputTree->SetBranchAddress("trackPDG", &trackPDG);
    inputTree->SetBranchAddress("trackPosX", &trackPosX);
    inputTree->SetBranchAddress("trackPosY", &trackPosY);
    inputTree->SetBranchAddress("trackPosZ", &trackPosZ);
    inputTree->SetBranchAddress("trackMomX", &trackMomX);
    inputTree->SetBranchAddress("trackMomY", &trackMomY);
    inputTree->SetBranchAddress("trackMomZ", &trackMomZ);
    inputTree->SetBranchAddress("trackKE", &trackKE);
    inputTree->SetBranchAddress("trackTime", &trackTime);
    inputTree->SetBranchAddress("trackProcess", &trackProcess);

    Long64_t nentries = inputTree->GetEntries();
    if (nentries == 0) {
        std::cout << "Información: El TTree 'output' en el archivo de entrada está vacío." << std::endl;
        outputFile->Close();
        inputFile->Close();
        delete outputFile;
        delete inputFile;
        return;
    }
    std::cout << "Procesando " << nentries << " eventos de: " << data_input_filename << std::endl;

    Int_t Nsel_feature_tree = 0;

    for (Long64_t jentry = 0; jentry < nentries; jentry++) {
        if (jentry > 0 && jentry % 10 == 0) { // Imprimir progreso menos frecuente
            std::cout << "  Procesando evento " << jentry << "/" << nentries << std::endl;
        }

        inputTree->GetEntry(jentry);

        // Inicializar variables para el evento actual
        Ec1 = 0.f; Ec2 = 0.f;
        Qmx = 0.f; QmxP = 0.f; QmxM = 0.f;
        tminP = 1000.f; tminM = 1000.f;
        dp = 0.f; Qmx2_val = 0.f;
        Qtot_p = 0.f; Qtot_m = 0.f;
        nhits_fired_sipm = 0;

        // Variables temporales para cálculos del evento actual
        float sum_c = 0.f;
        float sum_o = 0.f;
        float posX = 0.f; float posY = 0.f;
        float cmX = 0.f; float cmY = 0.f;
        float Qtot = 0.f;
        int imx = -1;
        int imxP = -1;
        int imxM = -1;
        int imx2 = -1;
        std::vector<float> x, y, z, time;
        std::vector<int> Qpe;

        if (!mcPMTNPE || !mcPMTID || !mcPEHitTime || !mcPEx || !mcPEy || !mcPEz) { continue; }

        // Distribucion espacial de la carga (para los .gif) - Alineado con clustering_completo.C
        TH2F *hr_zp_t1 = new TH2F("hr_zp_t1", "Plane z>0 Prompt signal", 90, -900., 900., 90, -900., 900.);
        TH2F *hr_zm_t1 = new TH2F("hr_zm_t1", "Plane z<0 Prompt signal", 90, -900., 900., 90, -900., 900.);
        std::cout << "Evento " << jentry << ": Histogramas creados" << std::endl;

        int ipe = 0; // Corresponds to ipe_cumulative in old V2, and ipe in clustering_completo
        int ih = 0;  // Explicit counter for SiPMs, as in clustering_completo

        for (std::vector<int>::iterator it = mcPMTNPE->begin(); it != mcPMTNPE->end(); ++it, ++ih) { // Loop estilo clustering_completo
            int current_sipm_npe = *it; // Numero de PEs en el SiPM actual
            int ipe_old = ipe;          // Indice del primer PE de este SiPM
            ipe += current_sipm_npe;    // Actualiza el indice acumulado de PEs

            if (current_sipm_npe == 0) continue;

            // Robust bounds check (conservado de extract_features_V2, adaptado)
            if (static_cast<size_t>(ipe) > mcPEx->size() ||
                static_cast<size_t>(ipe) > mcPEy->size() ||
                static_cast<size_t>(ipe) > mcPEz->size() ||
                static_cast<size_t>(ipe) > mcPEHitTime->size()) {
                std::cerr << "Advertencia Evento " << jentry << ": ipe (" << ipe
                          << ") excede el tamaño de los vectores de PE para SiPM ih=" << ih <<". Saltando SiPM." << std::endl;
                // Si ipe ya se incrementó, y este SiPM se salta, ipe_old para el *siguiente* SiPM será el ipe actual,
                // y el ipe para el siguiente SiPM se calculará a partir de este valor. Esto es problemático.
                // Para mantener la integridad de ipe, si saltamos, debemos revertir la adición de current_sipm_npe.
                ipe -= current_sipm_npe; // Revertir para que el próximo ipe_old sea correcto.
                continue;
            }

            int iq = 0;      // Flag para procesar el primer PE del SiPM (para llenar x,y,z)
            int Qpe_1 = 0;   // Contador de PE por SiPM (como en clustering_completo)
            float tmin = 1.0e9; // Tiempo del PE más rápido en el SiPM (como en clustering_completo)

            for (int j = ipe_old; j < ipe; ++j) { // Loop sobre los PEs del SiPM actual (j es el indice global del PE)
                float Htime = mcPEHitTime->at(j); // Tiempo del PE actual
                if (Htime < tmin) {
                    tmin = Htime;
                }

                Qpe_1++; // Contar PEs para este SiPM

                if (iq == 0) { // Para el primer PE del SiPM, guardar sus coordenadas en x, y, z
                    // int idfibre = mcPMTID->at(ih); // mcPMTID no se usa en las features de extract_features_V2, omitido por ahora
                    // sipm.push_back(idfibre); // No hay vector sipm en extract_features_V2
                    x.push_back(mcPEx->at(j));
                    y.push_back(mcPEy->at(j));
                    z.push_back(mcPEz->at(j));
                    iq++;
                }

                // Se guarda la distribución espacial de carga (para los .gif)
                float x_PE = mcPEx->at(j);
                float y_PE = mcPEy->at(j);
                float z_PE = mcPEz->at(j);

                if (z_PE > 0.) {
                    hr_zp_t1->Fill(x_PE, y_PE);
                    if (Nsel_feature_tree < 3) std::cout << "Llenando hr_zp con x=" << x_PE << ", y=" << y_PE << std::endl;
                } else if (z_PE < 0.) {
                    hr_zm_t1->Fill(x_PE, y_PE);
                    if (Nsel_feature_tree < 3) std::cout << "Llenando hr_zm con x=" << x_PE << ", y=" << y_PE << std::endl;
                }
            }

            time.push_back(tmin); // Guarda el tiempo del PE más rápido en un SiPM
            if (Qpe_1 > 0) {      // Solo guardar si el SiPM realmente tuvo PEs (ya cubierto por current_sipm_npe == 0 check)
                Qpe.push_back(Qpe_1); // Guarda el nº de PE de cada SiPM que recibe hits
            }
        }

        int nh = Qpe.size();
        if (nh == 0) { continue; }
        nhits_fired_sipm = nh;

        for (int j = 0; j < nh; j++) {
            if (Qpe.at(j) > Qmx) { Qmx = Qpe.at(j); imx = j; }
            if (z.at(j) > 0 && Qpe.at(j) > QmxP) { QmxP = Qpe.at(j); imxP = j; }
            if (z.at(j) < 0 && Qpe.at(j) > QmxM) { QmxM = Qpe.at(j); imxM = j; }
        }
        Qmx = (Qmx > 0) ? Qmx : 0;
        QmxP = (QmxP > 0) ? QmxP : 0;
        QmxM = (QmxM > 0) ? QmxM : 0;

        tminP = (imxP != -1) ? time.at(imxP) : 1000.f;
        tminM = (imxM != -1) ? time.at(imxM) : 1000.f;

        // Calcular umbral para el cluster 1 (asegurando Qmx > 1 para evitar log(0))
        float Dth_c1 = (Qmx > 1.0f) ? 25.f * TMath::Log(Qmx) : 0.f;
        if (Dth_c1 < 0.f) Dth_c1 = 0.f; // Asegurar que no sea negativo
        
        for (int j = 0; j < nh; j++) {
            float x_center_c1 = (imx != -1) ? x.at(imx) : 0.f;
            float y_center_c1 = (imx != -1) ? y.at(imx) : 0.f;
            float dist_to_c1 = TMath::Sqrt(TMath::Power(x_center_c1 - x.at(j), 2) + TMath::Power(y_center_c1 - y.at(j), 2));
            
            cmX += Qpe.at(j) * x.at(j);
            cmY += Qpe.at(j) * y.at(j);
            Qtot += Qpe.at(j);

            if (z.at(j) > 0) Qtot_p += Qpe.at(j);
            else if (z.at(j) < 0) Qtot_m += Qpe.at(j);

            if (imx != -1 && dist_to_c1 < Dth_c1) {
                posX += Qpe.at(j) * x.at(j);
                posY += Qpe.at(j) * y.at(j);
                sum_c += Qpe.at(j);
            } else {
                sum_o += Qpe.at(j);
                if (Qpe.at(j) > Qmx2_val) {
                    Qmx2_val = Qpe.at(j); imx2 = j;
                }
            }
        }
        Ec1 = sum_c;
        Ec2 = sum_o;
        Qtot_p = (Qtot_p > 0) ? Qtot_p : 0;
        Qtot_m = (Qtot_m > 0) ? Qtot_m : 0;

        if (imx2 != -1 && Qmx2_val > 1.0f) {
            float Dth_c2 = 25.f * TMath::Log(Qmx2_val);
            if (Dth_c2 < 0.f) Dth_c2 = 0.f;
            float x_center_c2 = x.at(imx2);
            float y_center_c2 = y.at(imx2);

            for (int j = 0; j < nh; j++) {
                if (j == imx2) continue;
                float x_center_c1 = (imx != -1) ? x.at(imx) : 0.f;
                float y_center_c1 = (imx != -1) ? y.at(imx) : 0.f;
                float dist_to_c1 = TMath::Sqrt(TMath::Power(x_center_c1 - x.at(j), 2) + TMath::Power(y_center_c1 - y.at(j), 2));

                if (imx == -1 || dist_to_c1 >= Dth_c1) {
                    float dist_to_c2 = TMath::Sqrt(TMath::Power(x_center_c2 - x.at(j), 2) + TMath::Power(y_center_c2 - y.at(j), 2));
                    if (dist_to_c2 < Dth_c2) {
                        Qmx2_val += Qpe.at(j);
                    }
                }
            }
        }

        // Calcular centros de masa
        if (sum_c > 0.f) { 
            posX /= sum_c; 
            posY /= sum_c; 
        } else {
            // Si no hay cluster principal, usar la posición del hit con máxima carga
            if (imx != -1) {
                posX = x.at(imx);
                posY = y.at(imx);
            } else {
                posX = 0.f;
                posY = 0.f;
            }
        }

        // Calcular centro de masa total
        if (Qtot > 0.f) { 
            cmX /= Qtot; 
            cmY /= Qtot; 
        } else { 
            cmX = 0.f; 
            cmY = 0.f;
        }
        
        // Calcular dp (distancia entre posicion del cluster 1 y centro de masa total)
        dp = TMath::Sqrt(TMath::Power(posX - cmX, 2) + TMath::Power(posY - cmY, 2));
        

        // Validar valores antes de calcular dp
        if (!std::isfinite(posX) || !std::isfinite(posY) || !std::isfinite(cmX) || !std::isfinite(cmY)) {
            std::cerr << "Advertencia: Valores no finitos en posiciones. Saltando evento." << std::endl;
            continue;
        }

        // Corte de volumen fiducial (en mm)
        if (cmX < -600 || cmX > 600 || cmY < -600 || cmY > 600) {
            std::cout << "Skipping entry " << jentry << ": center of mass outside fiducial volume. cmX = " << cmX << ", cmY = " << cmY << std::endl;
            delete hr_zp_t1;
            delete hr_zm_t1;
            continue;
        }

        // Generar GIF de la distribución espacial cada 10 eventos seleccionados
        // Comentado para evitar que se muestren las ventanas emergentes
        /*
        if ((Nsel_feature_tree + 1) % 10 == 0) {
            TCanvas *c1 = new TCanvas("c1", "Distribucion espacial de carga", 1200, 600);
            c1->Divide(2, 1);
            c1->cd(1);
            hr_zp_t1->SetStats(0);
            hr_zp_t1->Draw("COLZ");
            c1->cd(2);
            hr_zm_t1->SetStats(0);
            hr_zm_t1->Draw("COLZ");
            TString filename = TString::Format("distribucion_espacial_gif/evento_%d.gif", Nsel_feature_tree + 1);
            c1->SaveAs(filename);
            delete c1; // Liberar memoria del canvas
        }
        */

        // Guardar los histogramas en el archivo de salida
        // Usar nombres únicos para cada histograma
        TString histoNameZP = TString::Format("hr_zp_evt%d", Nsel_feature_tree);
        TString histoNameZM = TString::Format("hr_zm_evt%d", Nsel_feature_tree);
        
        // Renombrar los histogramas para evitar conflictos
        hr_zp_t1->SetName(histoNameZP);
        hr_zm_t1->SetName(histoNameZM);
        hr_zp_t1->SetTitle(histoNameZP);
        hr_zm_t1->SetTitle(histoNameZM);
        
        // Cambiar al directorio de histogramas y escribir los histogramas
        histoDir->cd();
        hr_zp_t1->Write();
        hr_zm_t1->Write();
        
        // Volver al directorio principal
        outputFile->cd();
        
        // Mostrar información de depuración
        std::cout << "Evento " << jentry << " - Entradas en " << histoNameZP << ": " << hr_zp_t1->GetEntries() << std::endl;
        std::cout << "Evento " << jentry << " - Entradas en " << histoNameZM << ": " << hr_zm_t1->GetEntries() << std::endl;
        
        // Borrar histogramas después de guardarlos
        delete hr_zp_t1;
        delete hr_zm_t1;
        
        // Llenar el árbol de características con los datos calculados
        event_number = Nsel_feature_tree + 1;
        mcpdg = mcpdg_input;  // Copiar el valor de mcpdg del árbol de entrada
        featureTree->Fill();
        Nsel_feature_tree++;

    } // Fin del bucle de eventos

    std::cout << "-----------------------------------------------------" << std::endl;
    std::cout << "Total de eventos en el archivo de entrada: " << nentries << std::endl;
    std::cout << "Total de eventos procesados y guardados: " << Nsel_feature_tree << std::endl;
    std::cout << "Archivo de salida: " << output_filename << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;

    // Escribir los árboles
    outputTree->Write("", TObject::kOverwrite);
    featureTree->Write();
    outputFile->Close();
    inputFile->Close();

    delete outputFile;
    delete inputFile;
}